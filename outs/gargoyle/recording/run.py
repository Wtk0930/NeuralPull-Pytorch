# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dataset import DatasetNP
from models.fields import NPullNetwork
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh
from models.utils import get_root_logger, print_log
import math
import mcubes
import warnings
from scipy.spatial import KDTree
warnings.filterwarnings("ignore")


class Runner:
    def __init__(self, args, conf_path, mode='train'):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.np_data_name'] = self.conf['dataset.np_data_name']
        self.base_exp_dir = self.conf['general.base_exp_dir'] + args.dir
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        
        self.dataset_np = DatasetNP(self.conf['dataset'], args.dataname)
        self.dataname = args.dataname
        self.iter_step = 0

        _, _, self.point_gt = self.dataset_np.np_train_data(1)
        self.kd_tree = KDTree(self.point_gt.detach().cpu().numpy())

        # Training parameters
        self.maxiter = self.conf.get_int('train.maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.eval_num_points = self.conf.get_int('train.eval_num_points')

        self.mode = mode

        # Networks
        self.sdf_network = NPullNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.optimizer = torch.optim.Adam(self.sdf_network.parameters(), lr=self.learning_rate)


        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()
    

    def compute_imls_surface(self, samples, k=10, sigma_r=0.1):
        """
        计算 samples 对应的 IMLS 表面点，使用提前构建的 KDTree
        :param samples: 查询点 (batch_size, num_samples, 3)
        :param k: 每个 sample 对应的 patch 中的邻近点数目
        :param sigma_r: 控制法向量一致性的高斯核参数
        :return: IMLS 表面点 (batch_size, num_samples, 3)
        """
        # 使用提前构建的 KDTree 查找 samples 的 k 个邻近点
        dist, indices = self.kd_tree.query(samples.detach().cpu().numpy(), k=k)

        # 获取邻近点 (5000, k, 3)
        neighbor_points = self.point_gt[indices]  # (5000, k, 3)

        # 对 samples 计算 SDF 和法向量
        gradients_sample = self.sdf_network.gradient(samples).squeeze().detach()  # (5000, 3)
        grad_norm = F.normalize(gradients_sample, dim=1)  # (5000, 3)

        # 计算邻近点的 SDF 和法向量 (5000, k, 3)
        neighbor_gradients = self.sdf_network.gradient(neighbor_points.view(-1, 3)).view(samples.shape[0], k, 3)
        neighbor_grad_norm = F.normalize(neighbor_gradients, dim=-1)  # (5000, k, 3)

        # 计算加权法向量一致性权重
        normal_proj_dist = torch.norm(grad_norm.unsqueeze(1) - neighbor_grad_norm, dim=-1) ** 2  # (5000, k)
        weight_phi = torch.exp(-normal_proj_dist / sigma_r ** 2)  # (5000, k)

        # 计算支持半径（support_radius）作为距离权重的尺度
        # bd_diag 是邻近点 patch 的对角线长度 (5000, k)
        bd_diag = torch.norm(neighbor_points.max(1)[0] - neighbor_points.min(1)[0], dim=-1, keepdim=True)  # (5000, 1)

        # 计算每个 patch 的支持半径 (5000, 1)
        support_radius = torch.sqrt(bd_diag / k)  # (5000, 1)

        # 计算距离权重，使用 support_radius 作为 sigma_dist 的替代 (5000, k)
        dist_sq = torch.tensor(dist ** 2, device=samples.device)  # (5000, k)
        weight_theta = torch.exp(-dist_sq / support_radius ** 2)  # (5000, k)

        # 计算最终加权 (5000, k)
        weight = weight_theta * weight_phi  # (5000, k)
        weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-12)  # (5000, k), 对每个 patch 的点归一化

        # 计算 IMLS 表面点 (5000, 3)
        imls_surface_points = (neighbor_points * weight.unsqueeze(-1)).sum(dim=1)  # (5000, 3)

        return imls_surface_points



    def train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        batch_size = self.batch_size

        res_step = self.maxiter - self.iter_step

        for iter_i in tqdm(range(res_step)):
            self.update_learning_rate_np(iter_i)

            # t_i q p_gt
            points, samples, point_gt = self.dataset_np.np_train_data(batch_size)

            
            samples.requires_grad = True
            gradients_sample = self.sdf_network.gradient(samples).squeeze() # 5000x3


            # get the sdf value -- the distance to the surface
            sdf_sample = self.sdf_network.sdf(samples)                      # 5000x1
            # get the gradient of the sdf
            grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3
            # move the sample points towards the surface
            sample_moved = samples - grad_norm * sdf_sample                 # 5000x3

            # Use the IMLS surface computation
            imls_surface_points = self.compute_imls_surface(samples, k=10, sigma_r=0.1)  # Compute IMLS surface points

            # loss function
            loss_sdf = torch.linalg.norm((imls_surface_points - sample_moved), ord=2, dim=-1).mean()
            
            loss = loss_sdf
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} cd_l1 = {} lr={}'.format(self.iter_step, loss_sdf, self.optimizer.param_groups[0]['lr']), logger=logger)

            if self.iter_step % self.val_freq == 0 and self.iter_step!=0: 
                self.validate_mesh(resolution=256, threshold=args.mcubes_threshold, point_gt=point_gt, iter_step=self.iter_step, logger=logger)

            if self.iter_step % self.save_freq == 0 and self.iter_step!=0: 
                self.save_checkpoint()


    def validate_mesh(self, resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None):

        bound_min = torch.tensor(self.dataset_np.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset_np.object_bbox_max, dtype=torch.float32)
        os.makedirs(os.path.join(self.base_exp_dir, 'outputs'), exist_ok=True)
        mesh = self.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, query_func=lambda pts: -self.sdf_network.sdf(pts))

        mesh.export(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}.ply'.format(self.iter_step,str(threshold))))


    def update_learning_rate_np(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def extract_fields(self, bound_min, bound_max, resolution, query_func):
        N = 32
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        return u

    def extract_geometry(self, bound_min, bound_max, resolution, threshold, query_func):
        print('Creating mesh with threshold: {}'.format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution, query_func)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        mesh = trimesh.Trimesh(vertices, triangles)

        return mesh

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        
        self.iter_step = checkpoint['iter_step']
            
    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    
        
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/np_srb.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcubes_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dir', type=str, default='gargoyle')
    parser.add_argument('--dataname', type=str, default='gargoyle')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args, args.conf, args.mode)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        threshs = [-0.001,-0.0025,-0.005,-0.01,-0.02,0.0,0.001,0.0025,0.005,0.01,0.02] 
        for thresh in threshs:
            runner.validate_mesh(resolution=256, threshold=thresh)