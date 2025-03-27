#!/usr/bin/env python3

import rospy
import numpy as np
import tf.transformations
from geometry_msgs.msg import PoseStamped

def generate_grasp_pose(obj_pose, frame_id="rgb_camera_link", height_offset=0.05):
    """基于物体位置生成抓取位姿"""
    grasp_pose = PoseStamped()
    grasp_pose.header.stamp = rospy.Time.now()
    grasp_pose.header.frame_id = frame_id
    
    # 设置位置（稍微在物体上方）
    grasp_pose.pose.position.x = obj_pose.position.x
    grasp_pose.pose.position.y = obj_pose.position.y
    grasp_pose.pose.position.z = obj_pose.position.z + height_offset  # 抓取位置略高于物体
    
    # 设置方向（向下抓取）
    q = tf.transformations.quaternion_from_euler(-np.pi/2, 0, 0)
    grasp_pose.pose.orientation.x = q[0]
    grasp_pose.pose.orientation.y = q[1]
    grasp_pose.pose.orientation.z = q[2]
    grasp_pose.pose.orientation.w = q[3]
    
    return grasp_pose

def compute_grasp_from_pointcloud(pcd, camera_matrix=None):
    """从点云计算抓取位姿"""
    try:
        from . import pointcloud_utils
        
        # 处理点云
        pcd, grasp_point, grasp_direction = pointcloud_utils.process_pointcloud(pcd)
        
        if grasp_point is None:
            return None, 0.0
            
        # 计算抓取评分
        compactness = 0.5  # 默认值
        size_score = min(1.0, len(np.asarray(pcd.points)) / 100.0)  # 归一化大小
        score = 0.5 * compactness + 0.5 * size_score
        
        # 创建抓取位姿
        grasp_pose = create_grasp_pose_from_point_and_direction(grasp_point, grasp_direction)
        
        return grasp_pose, score
    except Exception as e:
        rospy.logerr(f"点云抓取计算错误: {e}")
        return None, 0.0

def create_grasp_pose_from_point_and_direction(point, direction, frame_id="camera_color_optical_frame"):
    """从抓取点和方向创建PoseStamped消息"""
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = frame_id
    
    # 设置位置
    pose.pose.position.x = point[0]
    pose.pose.position.y = point[1]
    pose.pose.position.z = point[2]
    
    # 标准化方向向量
    direction = direction / np.linalg.norm(direction)
    
    # 创建一个从z轴到目标方向的旋转
    z_axis = np.array([0, 0, 1])
    
    # 计算旋转轴和角度
    rotation_axis = np.cross(z_axis, direction)
    
    if np.linalg.norm(rotation_axis) < 1e-6:
        # 向量平行，设置默认旋转轴
        rotation_axis = np.array([1, 0, 0])
        angle = 0 if direction[2] > 0 else np.pi
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.dot(z_axis, direction))
    
    # 轴角转四元数
    sin_half = np.sin(angle / 2)
    qx = rotation_axis[0] * sin_half
    qy = rotation_axis[1] * sin_half
    qz = rotation_axis[2] * sin_half
    qw = np.cos(angle / 2)
    
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    
    return pose

def create_default_grasp():
    """创建默认抓取位姿"""
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "camera_color_optical_frame"
    
    # 明显不同的默认位置
    pose.pose.position.x = 0.0
    pose.pose.position.y = 0.0
    pose.pose.position.z = 0.5
    
    # 默认方向（向下抓取）
    q = tf.transformations.quaternion_from_euler(-np.pi/2, 0, 0)
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    
    return pose