#!/usr/bin/env python3

import rospy
import numpy as np

def depth_to_pointcloud(rgb, depth, camera_matrix):
    """Convert RGB-D image to colored point cloud"""
    try:
        import open3d as o3d
        
        # Get camera intrinsics
        height, width = depth.shape
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        
        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        
        # Create mesh grid coordinates
        x = np.arange(0, width)
        y = np.arange(0, height)
        xx, yy = np.meshgrid(x, y)
        
        # Get indices of valid depth values
        valid = depth > 0.001
        
        # Calculate 3D points
        z = depth[valid]
        x = (xx[valid] - cx) * z / fx
        y = (yy[valid] - cy) * z / fy
        
        # Concatenate XYZ coordinates
        points = np.vstack((x, y, z)).T
        
        # Get corresponding RGB values
        if len(rgb.shape) == 3 and rgb.shape[2] == 3:
            colors = rgb[valid] / 255.0  # Normalize to [0,1]
        else:
            colors = np.ones((points.shape[0], 3)) * 0.5  # Gray
        
        # Set points and colors for the point cloud
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    except ImportError:
        rospy.logwarn("Cannot import Open3D, point cloud processing unavailable")
        return None
    except Exception as e:
        rospy.logerr(f"Point cloud conversion error: {e}")
        return None

def process_pointcloud(pcd):
    """Process point cloud: down-sampling, compute normals, remove outliers"""
    try:
        import open3d as o3d
        
        if pcd is None or len(np.asarray(pcd.points)) < 10:
            rospy.logwarn("Too few points in point cloud for processing")
            return None, None, None
            
        # Point cloud processing: down-sampling, remove outliers
        # MODIFY: Adjust voxel_size based on your scene scale and required detail level
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        # MODIFY: Adjust outlier removal parameters based on your data quality
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)
        
        # Estimate normals
        # MODIFY: Adjust normal estimation parameters based on your point cloud density
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        
        # Get point cloud data
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        if len(points) < 10:
            rospy.logwarn("Too few points after processing")
            return None, None, None
        
        # Calculate principal direction of point cloud (using PCA)
        mean = np.mean(points, axis=0)
        points_centered = points - mean
        cov = points_centered.T @ points_centered / len(points)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Principal axis is the eigenvector corresponding to largest eigenvalue
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Ensure principal axis points upward
        if principal_axis[2] < 0:
            principal_axis = -principal_axis
            
        return pcd, mean, principal_axis
    except Exception as e:
        rospy.logerr(f"Point cloud processing error: {e}")
        return None, None, None

def prepare_pointcloud_for_graspnet(depth, rgb, camera_matrix):
    """Prepare point cloud format required by GraspNet"""
    try:
        import open3d as o3d
        
        # Create point cloud
        pcd = depth_to_pointcloud(rgb, depth, camera_matrix)
        if pcd is None or len(np.asarray(pcd.points)) < 10:
            rospy.logwarn("Too few points in point cloud")
            return None
            
        # Down-sampling
        # MODIFY: Adjust voxel_size based on your requirements and computational capabilities
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        
        # Remove outliers
        # MODIFY: Adjust outlier removal parameters based on your data quality
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return pcd
    except ImportError:
        rospy.logerr("Missing open3d, cannot process point cloud")
        return None
    except Exception as e:
        rospy.logerr(f"Failed to prepare point cloud: {e}")
        return None