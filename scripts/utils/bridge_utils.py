#!/usr/bin/env python3
# filepath: d:\GitHub repo\yolov8_graspnet_ros\yolo_graspnet_ros\scripts\utils\bridge_utils.py

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image

def get_bridge():
    """Get image conversion bridge (cv_bridge or custom implementation)"""
    try:
        from cv_bridge import CvBridge
        bridge = CvBridge()
        rospy.loginfo("Successfully imported cv_bridge")
        return bridge
    except ImportError:
        rospy.logwarn("Unable to import cv_bridge, using custom alternative")
        return CvBridgeLite()

class CvBridgeLite:
    """Simplified cv_bridge, implements only basic image conversion functions"""
    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):
        if desired_encoding == "passthrough":
            encoding = img_msg.encoding
        else:
            encoding = desired_encoding
        
        # Determine data type and channel count based on encoding
        if encoding in ['bgr8', 'rgb8']:
            dtype, n_channels = np.uint8, 3
        elif encoding == 'mono8':
            dtype, n_channels = np.uint8, 1
        elif encoding == 'mono16':
            dtype, n_channels = np.uint16, 1
        elif encoding in ['32FC1', 'passthrough'] and img_msg.encoding == '32FC1':
            dtype, n_channels = np.float32, 1
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")
        
        # Convert byte data to numpy array
        if n_channels == 1:
            im = np.frombuffer(img_msg.data, dtype=dtype).reshape(
                img_msg.height, img_msg.width)
        else:
            im = np.frombuffer(img_msg.data, dtype=dtype).reshape(
                img_msg.height, img_msg.width, n_channels)
            
        # Color space conversion
        if desired_encoding == "bgr8" and img_msg.encoding == "rgb8":
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        elif desired_encoding == "rgb8" and img_msg.encoding == "bgr8":
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
        return im