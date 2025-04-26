#!/usr/bin/env python3

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

def draw_bounding_boxes(image, boxes, scores, classes, class_names):
    for box, score, cls in zip(boxes, scores, classes):
        # ⚠️ MODIFY: Adjust confidence threshold based on your application
        if score < 0.5:  # Threshold for displaying boxes
            continue
        x1, y1, x2, y2 = box
        # ⚠️ OPTIONAL: Customize colors for different classes
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
    """Create a visualization marker for grasp pose"""
    marker = Marker()
    # MODIFY: Change frame_id to match your camera frame
    marker.header.frame_id = "camera_color_optical_frame"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "grasp_markers"
    marker.id = id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    
    # Set position
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = float(position[2])
    
    # Calculate quaternion from rotation matrix
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
        # Use default orientation if conversion fails
        marker.pose.orientation.w = 1.0
    
    # Set arrow dimensions
    marker.scale.x = width  # Length
    marker.scale.y = 0.01   # Width
    marker.scale.z = 0.01   # Height
    
    # Set color based on score
    marker.color.r = 1.0 - score  # More red when score is low
    marker.color.g = score        # More green when score is high
    marker.color.b = 0.0
    marker.color.a = 0.7
    
    # Set lifetime
    marker.lifetime = rospy.Duration(2.0)  # 2 seconds
    
    return marker

def create_marker_array(pose_list, scores=None):
    """Create an array of markers from multiple poses"""
    marker_array = MarkerArray()
    
    for i, pose in enumerate(pose_list):
        score = scores[i] if scores is not None and i < len(scores) else 0.5
        marker = create_grasp_marker(pose, score, i)
        marker_array.markers.append(marker)
    
    return marker_array

def create_grasp_poses_marker_array(grasp_results, frame_id="camera_color_optical_frame"):
    """Create marker array from GraspNet results"""
    from visualization_msgs.msg import MarkerArray
    from scipy.spatial.transform import Rotation
    from geometry_msgs.msg import Pose
    
    marker_array = MarkerArray()
    
    # MODIFY: Adjust number of displayed grasps based on your needs
    for i, grasp in enumerate(grasp_results[:20]):  # Only display top 20
        pose = Pose()
        pose.position.x = grasp['point'][0]
        pose.position.y = grasp['point'][1]
        pose.position.z = grasp['point'][2]
        
        # Convert rotation matrix to quaternion
        r = Rotation.from_matrix(grasp['rotation'])
        quat = r.as_quat()  # [x, y, z, w]
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        
        # Create marker
        marker = create_grasp_marker(pose, grasp['score'], f"grasp", i)
        marker.header.frame_id = frame_id
        marker_array.markers.append(marker)
    
    return marker_array