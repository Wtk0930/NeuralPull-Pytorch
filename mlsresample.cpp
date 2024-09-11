#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>

int main ()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    if (pcl::io::loadPLYFile<pcl::PointXYZ>("./data/gargoyle.ply", *cloud) == -1) 
    {
        PCL_ERROR("Couldn't read file ./data/gargoyle.ply \n");
        return (-1);
    }

    std::cout << "Loaded " << cloud->size() << " data points from gargoyle.ply" << std::endl;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal>::Ptr mls_points (new pcl::PointCloud<pcl::PointNormal>);

    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    mls.setComputeNormals (true);
    mls.setInputCloud (cloud);
    mls.setSearchMethod (tree);
    
    int orders[] = {1, 2};
    double radii[] = {0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0, 1,1, 1.2, 1.5, 2.0};

    for (int order : orders) {
        for (double radius : radii) {
            mls.setPolynomialOrder(order);
            mls.setSearchRadius(radius);
            mls.process(*mls_points);

            std::cout << "Polynomial Order: " << order << ", Radius: " << radius << std::endl;
            std::cout << "Output has " << mls_points->size() << " data points." << std::endl;
        }
    }
  
    return 0;
}
