#!/usr/bin/env python3

import rospy
import numpy as np
import tf.transformations
from geometry_msgs.msg import PoseStamped

def generate_grasp_pose(obj_pose, frame_id="rgb_camera_link", height_offset=0.05):
    """Generate grasp pose based on object position"""
    grasp_pose = PoseStamped()
    grasp_pose.header.stamp = rospy.Time.now()
    # MODIFY: Change frame_id to match your camera/robot frame
    grasp_pose.header.frame_id = frame_id
    
    # Set position (slightly above the object)
    grasp_pose.pose.position.x = obj_pose.position.x
    grasp_pose.pose.position.y = obj_pose.position.y
    grasp_pose.pose.position.z = obj_pose.position.z + height_offset  # Grasp position slightly above the object
    
    # Set orientation (downward grasp)
    q = tf.transformations.quaternion_from_euler(-np.pi/2, 0, 0)
    grasp_pose.pose.orientation.x = q[0]
    grasp_pose.pose.orientation.y = q[1]
    grasp_pose.pose.orientation.z = q[2]
    grasp_pose.pose.orientation.w = q[3]
    
    return grasp_pose

def compute_grasp_from_pointcloud(pcd, camera_matrix=None):
    """Compute grasp pose from point cloud"""
    try:
        from . import pointcloud_utils
        
        # Process point cloud
        pcd, grasp_point, grasp_direction = pointcloud_utils.process_pointcloud(pcd)
        
        if grasp_point is None:
            return None, 0.0
            
        # Calculate grasp score
        compactness = 0.5  # Default value
        size_score = min(1.0, len(np.asarray(pcd.points)) / 100.0)  # Normalized size
        score = 0.5 * compactness + 0.5 * size_score
        
        # Create grasp pose
        grasp_pose = create_grasp_pose_from_point_and_direction(grasp_point, grasp_direction)
        
        return grasp_pose, score
    except Exception as e:
        rospy.logerr(f"Point cloud grasp calculation error: {e}")
        return None, 0.0

def create_grasp_pose_from_point_and_direction(point, direction, frame_id="camera_color_optical_frame"):
    """Create PoseStamped message from grasp point and direction"""
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    # MODIFY: Change frame_id to match your camera frame
    pose.header.frame_id = frame_id
    
    # Set position
    pose.pose.position.x = point[0]
    pose.pose.position.y = point[1]
    pose.pose.position.z = point[2]
    
    # Normalize direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Create a rotation from z-axis to target direction
    z_axis = np.array([0, 0, 1])
    
    # Calculate rotation axis and angle
    rotation_axis = np.cross(z_axis, direction)
    
    if np.linalg.norm(rotation_axis) < 1e-6:
        # Vectors are parallel, set default rotation axis
        rotation_axis = np.array([1, 0, 0])
        angle = 0 if direction[2] > 0 else np.pi
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.dot(z_axis, direction))
    
    # Axis-angle to quaternion
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
    """Create default grasp pose"""
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    # MODIFY: Change frame_id to match your camera frame
    pose.header.frame_id = "camera_color_optical_frame"
    
    # Distinctly different default position
    pose.pose.position.x = 0.0
    pose.pose.position.y = 0.0
    pose.pose.position.z = 0.5
    
    # Default orientation (downward grasp)
    q = tf.transformations.quaternion_from_euler(-np.pi/2, 0, 0)
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    
    return pose