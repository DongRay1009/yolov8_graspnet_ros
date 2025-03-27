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
        
        # Get indices with valid depth values
        valid = depth > 0.001
        
        # Calculate 3D points
        z = depth[valid]
        x = (xx[valid] - cx) * z / fx
        y = (yy[valid] - cy) * z / fy
        
        # Combine XYZ coordinates
        points = np.vstack((x, y, z)).T
        
        # Get corresponding RGB values
        if len(rgb.shape) == 3 and rgb.shape[2] == 3:
            colors = rgb[valid] / 255.0  # Normalize to [0,1]
        else:
            colors = np.ones((points.shape[0], 3)) * 0.5  # Gray color
        
        # Set point cloud points and colors
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    except ImportError:
        rospy.logwarn("Unable to import Open3D, point cloud processing unavailable")
        return None
    except Exception as e:
        rospy.logerr(f"Point cloud conversion error: {e}")
        return None

def process_pointcloud(pcd):
    """Process point cloud: downsample, compute normals, remove outliers"""
    try:
        import open3d as o3d
        
        if pcd is None or len(np.asarray(pcd.points)) < 10:
            rospy.logwarn("Too few points in point cloud for processing")
            return None, None, None
            
        # Point cloud processing: downsample, remove outliers
        # MODIFY: Adjust voxel_size parameter based on your point cloud density
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)
        
        # Estimate normals
        # MODIFY: Adjust search radius and max_nn parameters for normal estimation based on your point cloud density
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        
        # Get point cloud data
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        if len(points) < 10:
            rospy.logwarn("Too few points in processed point cloud")
            return None, None, None
        
        # Calculate principal direction (using PCA)
        mean = np.mean(points, axis=0)
        points_centered = points - mean
        cov = points_centered.T @ points_centered / len(points)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Principal axis is the eigenvector corresponding to the largest eigenvalue
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Ensure principal axis points upward
        if principal_axis[2] < 0:
            principal_axis = -principal_axis
            
        return pcd, mean, principal_axis
    except Exception as e:
        rospy.logerr(f"Point cloud processing error: {e}")
        return None, None, None

def prepare_pointcloud_for_graspnet(depth, rgb, camera_matrix):
    """Prepare point cloud in the format required by GraspNet"""
    try:
        import open3d as o3d
        
        # Create point cloud
        pcd = depth_to_pointcloud(rgb, depth, camera_matrix)
        if pcd is None or len(np.asarray(pcd.points)) < 10:
            rospy.logwarn("Too few points in point cloud")
            return None
            
        # Downsample
        # MODIFY: Adjust voxel_size parameter based on your point cloud density
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        
        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return pcd
    except ImportError:
        rospy.logerr("Missing open3d, unable to process point cloud")
        return None
    except Exception as e:
        rospy.logerr(f"Failed to prepare point cloud: {e}")
        return None

def create_object_point_cloud(depth_image, camera_info, x_center, y_center, radius=50):
    """Create object point cloud based on detection box"""
    height, width = depth_image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create circular region
    import cv2
    # MODIFY: Adjust radius parameter based on your object size
    cv2.circle(mask, (int(x_center), int(y_center)), radius, 255, -1)
    
    # Create coordinate grid
    v, u = np.mgrid[0:height, 0:width]
    
    # Get valid region
    valid_mask = (mask > 0) & (depth_image > 0)
    valid_points = np.where(valid_mask)
    
    if len(valid_points[0]) < 10:
        rospy.logwarn(f"Too few valid points in object region: {len(valid_points[0])}")
        return np.zeros((0, 3))
    
    # Get valid depth values
    z = depth_image[valid_points].astype(np.float32) / 1000.0  # Convert to meters
    
    # Back-project to 3D space
    fx = camera_info.fx if hasattr(camera_info, 'fx') else camera_info[0, 0]
    fy = camera_info.fy if hasattr(camera_info, 'fy') else camera_info[1, 1]
    cx = camera_info.cx if hasattr(camera_info, 'cx') else camera_info[0, 2]
    cy = camera_info.cy if hasattr(camera_info, 'cy') else camera_info[1, 2]
    
    x = (u[valid_points] - cx) * z / fx
    y = (v[valid_points] - cy) * z / fy
    
    # Combine into point cloud
    point_cloud = np.stack([x, y, z], axis=1)
    
    return point_cloud

def create_pose_from_grasp(grasp):
    """Convert grasp to ROS pose"""
    from geometry_msgs.msg import Pose
    import tf.transformations
    
    pose = Pose()
    
    # Set position
    pose.position.x = float(grasp['position'][0]) if isinstance(grasp['position'], np.ndarray) else float(grasp['position'].x)
    pose.position.y = float(grasp['position'][1]) if isinstance(grasp['position'], np.ndarray) else float(grasp['position'].y)
    pose.position.z = float(grasp['position'][2]) if isinstance(grasp['position'], np.ndarray) else float(grasp['position'].z)
    
    # Set orientation
    rot_matrix = grasp.get('rotation')
    if rot_matrix is not None and hasattr(rot_matrix, 'shape') and rot_matrix.shape == (3, 3):
        # Create 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_matrix
        
        # Convert to quaternion
        q = tf.transformations.quaternion_from_matrix(transform_matrix)
        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])
    else:
        # Default orientation (downward)
        pose.orientation.w = 1.0
    
    return pose