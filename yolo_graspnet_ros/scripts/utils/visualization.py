#!/usr/bin/env python3
# filepath: /home/msi/yolo_3d_ws/src/yolo_graspnet_ros/scripts/utils/visualization.py

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

def draw_bounding_boxes(image, boxes, scores, classes, class_names):
    for box, score, cls in zip(boxes, scores, classes):
        # ⚠️ 需要调整: 根据应用场景可能需要调整置信度阈值
        if score < 0.5:  # Threshold for displaying boxes
            continue
        x1, y1, x2, y2 = box
        # ⚠️ 可选调整: 可以自定义不同类别的边界框颜色
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_names[cls]}: {score:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def visualize_detections(image, detections, class_names):
    boxes = detections['boxes']
    scores = detections['scores']
    classes = detections['classes']
    return draw_bounding_boxes(image, boxes, scores, classes, class_names)

def create_grasp_marker(position, rotation, width, score, id=0):
    """创建抓取可视化标记"""
    marker = Marker()
    marker.header.frame_id = "camera_color_optical_frame"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "grasp_markers"
    marker.id = id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    
    # 设置位置
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = float(position[2])
    
    # 从旋转矩阵计算四元数
    try:
        import tf.transformations
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        q = tf.transformations.quaternion_from_matrix(matrix)
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
    except:
        # 如果转换失败，使用默认方向
        marker.pose.orientation.w = 1.0
    
    # 设置箭头尺寸
    marker.scale.x = width  # 长度
    marker.scale.y = 0.01   # 宽度
    marker.scale.z = 0.01   # 高度
    
    # 根据评分设置颜色
    marker.color.r = 1.0 - score  # 分数低时更红
    marker.color.g = score        # 分数高时更绿
    marker.color.b = 0.0
    marker.color.a = 0.7
    
    # 设置生命周期
    marker.lifetime = rospy.Duration(2.0)  # 2秒
    
    return marker

def create_marker_array(pose_list, scores=None):
    """创建多个标记的数组"""
    marker_array = MarkerArray()
    
    for i, pose in enumerate(pose_list):
        score = scores[i] if scores is not None and i < len(scores) else 0.5
        marker = create_grasp_marker(pose, score, i)
        marker_array.markers.append(marker)
    
    return marker_array

def create_grasp_poses_marker_array(grasp_results, frame_id="camera_color_optical_frame"):
    """从GraspNet结果创建标记数组"""
    from visualization_msgs.msg import MarkerArray
    from scipy.spatial.transform import Rotation
    from geometry_msgs.msg import Pose
    
    marker_array = MarkerArray()
    
    for i, grasp in enumerate(grasp_results[:20]):  # 只显示前20个
        pose = Pose()
        pose.position.x = grasp['point'][0]
        pose.position.y = grasp['point'][1]
        pose.position.z = grasp['point'][2]
        
        # 转换旋转矩阵为四元数
        r = Rotation.from_matrix(grasp['rotation'])
        quat = r.as_quat()  # [x, y, z, w]
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        
        # 创建标记
        marker = create_grasp_marker(pose, grasp['score'], f"grasp", i)
        marker.header.frame_id = frame_id
        marker_array.markers.append(marker)
    
    return marker_array