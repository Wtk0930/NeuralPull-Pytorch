# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from models.dataset import DatasetNP

from models.train_dataset import PointGenerateDataset, SequentialPointCloudRandomPatchSampler
from models.fields import NPullNetwork
import torch.nn as nn
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
from torch.utils import data
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



    def compute_imls_surface(self, samples,neighbor_points, dist, k=10, sigma_r=0.1):
        """
        计算 samples 对应的 IMLS 表面点，使用提前构建的 KDTree
        :param samples: 查询点 (batch_size, num_samples, 3)
        :param k: 每个 sample 对应的 patch 中的邻近点数目
        :param sigma_r: 控制法向量一致性的高斯核参数
        :return: IMLS 表面点 (batch_size, num_samples, 3)
        """

        # 对 samples 计算 SDF 和法向量
        gradients_sample = self.sdf_network.gradient(samples).squeeze().detach()  # (5000, 3)
        grad_norm = F.normalize(gradients_sample, dim=1)  # (5000, 3)

        # 计算邻近点的 SDF 和法向量 (5000, k, 3)
        neighbor_gradients = self.sdf_network.gradient(neighbor_points.view(-1, 3)).detach().view(samples.shape[0], k, 3)
        neighbor_grad_norm = F.normalize(neighbor_gradients, dim=-1)  # (5000, k, 3)

        # 计算加权法向量一致性权重
        normal_proj_dist = torch.norm(grad_norm.unsqueeze(1) - neighbor_grad_norm, dim=-1) ** 2  # (5000, k)
        weight_phi = torch.exp(-normal_proj_dist / sigma_r ** 2)  # (5000, k)

        # 计算支持半径（support_radius）作为距离权重的尺度
        # bd_diag 是邻近点 patch 的对角线长度 (5000, k)
        bd_diag = torch.norm(neighbor_points.max(1)[0] - neighbor_points.min(1)[0], dim=-1, keepdim=True)  # (5000, 1)

        # 计算每个 patch 的支持半径 (5000, 1)
        support_radius = torch.sqrt(bd_diag / k**2)  # (5000, 1)
        # print(support_radius)

        # 计算距离权重，使用 support_radius 作为 sigma_dist 的替代 (5000, k)
        dist_sq = torch.tensor(dist ** 2, device=samples.device)  # (5000, k)
        weight_theta = torch.exp(-dist_sq / support_radius ** 2)  # (5000, k)

        # 计算最终加权 (5000, k)
        weight = weight_theta * weight_phi  # (5000, k)
        weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-12)  # (5000, k), 对每个 patch 的点归一化

        # 计算 IMLS 表面点 (5000, 3)
        project_dist = ((samples.unsqueeze(1) - neighbor_points) * neighbor_grad_norm).sum(2)  # (5000, k)

        # 计算加权的IMLS距离
        imls_dist = (project_dist * weight).sum(1, keepdim=True)  # (5000, 1)

        return imls_dist


    def train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        batch_size = self.batch_size

        res_step = self.maxiter - self.iter_step

        criterion = nn.MSELoss().to(args.gpu)



        train_ds = PointGenerateDataset("data/noisy_0.4.ply", gt_pts_num=-1, batch_size=100,
                                        query_num=None, patch_radius=0.01,
                                        points_per_patch=50, device="cuda")
        
        train_sampler = SequentialPointCloudRandomPatchSampler(data_source=train_ds, shape_num=train_ds.shape_num)
        train_dl = data.DataLoader(train_ds, batch_size=1, 
                               sampler=train_sampler,
                               )


        for epoch in trange(1, 5):

            self.sdf_network.train()
            with tqdm(total=len(train_dl), desc='train loop') as tq:
                for i, (t, q, lambda_p, proxy) in enumerate(train_dl):

                    self.update_learning_rate_np(self.iter_step)

                    # t is the patch of the query point 500 is the batch size
                    t = t.to(args.gpu, non_blocking=True)  # (1, 500, k, 3)

                    # q is the query point
                    q = q.to(args.gpu, non_blocking=True)  # (1, 500, 3)

                    # proxy is the nearest point of the query point
                    proxy = proxy.to(args.gpu, non_blocking=True)  # (1, 500, 3)

                    # lambda_p is the radius of the query point
                    lambda_p = lambda_p.to(args.gpu, non_blocking=True)  # (1, 500, 1)
                    lambda_p *= 1

                    t = t.squeeze(0)          # (500, k, 3)
                    q = q.squeeze(0)          # (500, 3)
                    lambda_p = lambda_p.squeeze(0)  # (500, 1)
                    proxy = proxy.squeeze(0)   # (500, 3)

                    t.requires_grad = True
                    q.requires_grad = True

                    # compute query sdf, normal
                    sdf = self.sdf_network.sdf(q)
                    sdf.sum().backward(retain_graph=True)
                    # the gradient of the query point
                    q_grad = q.grad.detach()  # (500, 3)
                    q_grad = F.normalize(q_grad, dim=-1)

                    # compute neighbor sdf, normal
                    neigh_sdf = self.sdf_network.sdf(t.reshape(-1, 3))
                    neigh_sdf.sum().backward(retain_graph=True)
                    neigh_grad = t.grad.detach()
                    neigh_grad = F.normalize(neigh_grad, dim=-1)

                    # IMLS
                    x = q  # (5000, 3)
                    dist = torch.linalg.norm(x.unsqueeze(1) - t, dim=-1) + 1e-8
                    dist_sq = dist ** 2
                    # Gaussian kernel:[images/image.png]
                    weight_theta = torch.exp(-dist_sq / lambda_p ** 2)  # (1, 5000, k)


                    ## RIMLS 's Style Gaussian Kernel for Normal [images/image2.png]
                    normal_proj_dist = torch.norm(q_grad.unsqueeze(1) - neigh_grad, dim=-1) ** 2
                    weight_phi = torch.exp(-normal_proj_dist / 0.3 ** 2)

                    # bilateral normal smooth [images/image3.png]
                    weight = weight_theta * weight_phi + 1e-12
                    weight = weight / weight.sum(1, keepdim=True)

                    project_dist = ((x.unsqueeze(1) - t) * neigh_grad).sum(-1)
                    imls_dist = (project_dist * weight).sum(1, keepdim=True)
                    # [images/image5.png]
                    loss = criterion(imls_dist, sdf)

                    q_moved = q.detach() - q_grad * sdf.detach()

                    q_moved.requires_grad = True
                    q_moved_sdf = self.sdf_network.sdf(q_moved)
                    q_moved_sdf.sum().backward(retain_graph=True)
                    q_moved_grad = q_moved.grad.detach()  # (1, 500, 3)
                    q_moved_grad = F.normalize(q_moved_grad, dim=-1)

                    consis_constraint = 1 - F.cosine_similarity(q_moved_grad, q_grad, dim=-1)
                    weight_moved = torch.exp(-10.0 * torch.abs(sdf)).reshape(-1, consis_constraint.shape[-1])
                    consis_constraint = (weight_moved * consis_constraint).mean()
                    
                    loss = loss + 0.01 * consis_constraint
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    self.iter_step += 1

                    if self.iter_step % self.report_freq == 0:
                        print_log('iter:{:8>d} cd_l1 = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']), logger=logger)

                    if self.iter_step % self.val_freq == 0 and self.iter_step!=0: 
                        self.validate_mesh(resolution=256, threshold=args.mcubes_threshold, point_gt=self.point_gt, iter_step=self.iter_step, logger=logger)

                    if self.iter_step % self.save_freq == 0 and self.iter_step!=0: 
                        self.save_checkpoint()
                    
                    tq.update(1)


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