#!/usr/bin/env python3

import rospy
import numpy as np

def depth_to_pointcloud(rgb, depth, camera_matrix):
    """将RGB-D图像转换为彩色点云"""
    try:
        import open3d as o3d
        
        # 获取相机内参
        height, width = depth.shape
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        
        # 创建网格点坐标
        x = np.arange(0, width)
        y = np.arange(0, height)
        xx, yy = np.meshgrid(x, y)
        
        # 获取有效深度值的索引
        valid = depth > 0.001
        
        # 计算3D点
        z = depth[valid]
        x = (xx[valid] - cx) * z / fx
        y = (yy[valid] - cy) * z / fy
        
        # 拼接XYZ坐标
        points = np.vstack((x, y, z)).T
        
        # 获取对应的RGB值
        if len(rgb.shape) == 3 and rgb.shape[2] == 3:
            colors = rgb[valid] / 255.0  # 归一化到[0,1]
        else:
            colors = np.ones((points.shape[0], 3)) * 0.5  # 灰色
        
        # 设置点云的点和颜色
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    except ImportError:
        rospy.logwarn("无法导入Open3D，点云处理不可用")
        return None
    except Exception as e:
        rospy.logerr(f"点云转换错误: {e}")
        return None

def process_pointcloud(pcd):
    """处理点云：降采样、计算法线、去除离群点"""
    try:
        import open3d as o3d
        
        if pcd is None or len(np.asarray(pcd.points)) < 10:
            rospy.logwarn("点云点数太少，无法处理")
            return None, None, None
            
        # 点云处理：降采样、去除离群点
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)
        
        # 估计法线
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        
        # 获取点云数据
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        if len(points) < 10:
            rospy.logwarn("处理后的点云点数太少")
            return None, None, None
        
        # 计算点云的主方向（使用PCA）
        mean = np.mean(points, axis=0)
        points_centered = points - mean
        cov = points_centered.T @ points_centered / len(points)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 主轴是对应于最大特征值的特征向量
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
        
        # 确保主轴指向上方
        if principal_axis[2] < 0:
            principal_axis = -principal_axis
            
        return pcd, mean, principal_axis
    except Exception as e:
        rospy.logerr(f"点云处理错误: {e}")
        return None, None, None

def prepare_pointcloud_for_graspnet(depth, rgb, camera_matrix):
    """准备GraspNet所需的点云格式"""
    try:
        import open3d as o3d
        
        # 创建点云
        pcd = depth_to_pointcloud(rgb, depth, camera_matrix)
        if pcd is None or len(np.asarray(pcd.points)) < 10:
            rospy.logwarn("点云点数太少")
            return None
            
        # 降采样
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        
        # 移除离群点
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return pcd
    except ImportError:
        rospy.logerr("缺少open3d，无法处理点云")
        return None
    except Exception as e:
        rospy.logerr(f"准备点云失败: {e}")
        return None

def create_object_point_cloud(depth_image, camera_info, x_center, y_center, radius=50):
    """根据检测框创建物体点云"""
    height, width = depth_image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 创建圆形区域
    import cv2
    cv2.circle(mask, (int(x_center), int(y_center)), radius, 255, -1)
    
    # 创建坐标网格
    v, u = np.mgrid[0:height, 0:width]
    
    # 获取有效区域
    valid_mask = (mask > 0) & (depth_image > 0)
    valid_points = np.where(valid_mask)
    
    if len(valid_points[0]) < 10:
        rospy.logwarn(f"物体区域内有效点太少: {len(valid_points[0])}")
        return np.zeros((0, 3))
    
    # 获取有效深度值
    z = depth_image[valid_points].astype(np.float32) / 1000.0  # 转换为米
    
    # 反投影到3D空间
    fx = camera_info.fx if hasattr(camera_info, 'fx') else camera_info[0, 0]
    fy = camera_info.fy if hasattr(camera_info, 'fy') else camera_info[1, 1]
    cx = camera_info.cx if hasattr(camera_info, 'cx') else camera_info[0, 2]
    cy = camera_info.cy if hasattr(camera_info, 'cy') else camera_info[1, 2]
    
    x = (u[valid_points] - cx) * z / fx
    y = (v[valid_points] - cy) * z / fy
    
    # 组合为点云
    point_cloud = np.stack([x, y, z], axis=1)
    
    return point_cloud

def create_pose_from_grasp(grasp):
    """将抓取转换为ROS位姿"""
    from geometry_msgs.msg import Pose
    import tf.transformations
    
    pose = Pose()
    
    # 设置位置
    pose.position.x = float(grasp['position'][0]) if isinstance(grasp['position'], np.ndarray) else float(grasp['position'].x)
    pose.position.y = float(grasp['position'][1]) if isinstance(grasp['position'], np.ndarray) else float(grasp['position'].y)
    pose.position.z = float(grasp['position'][2]) if isinstance(grasp['position'], np.ndarray) else float(grasp['position'].z)
    
    # 设置方向
    rot_matrix = grasp.get('rotation')
    if rot_matrix is not None and hasattr(rot_matrix, 'shape') and rot_matrix.shape == (3, 3):
        # 创建4x4变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_matrix
        
        # 转换为四元数
        q = tf.transformations.quaternion_from_matrix(transform_matrix)
        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])
    else:
        # 默认方向（向下）
        pose.orientation.w = 1.0
    
    return pose