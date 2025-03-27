#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image

def get_bridge():
    """获取图像转换桥（cv_bridge或自定义实现）"""
    try:
        from cv_bridge import CvBridge
        bridge = CvBridge()
        rospy.loginfo("成功导入cv_bridge")
        return bridge
    except ImportError:
        rospy.logwarn("无法导入cv_bridge，使用自定义替代方案")
        return CvBridgeLite()

class CvBridgeLite:
    """简化版cv_bridge，仅实现基本图像转换功能"""
    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):
        if desired_encoding == "passthrough":
            encoding = img_msg.encoding
        else:
            encoding = desired_encoding
        
        # 根据编码确定数据类型和通道数
        if encoding in ['bgr8', 'rgb8']:
            dtype, n_channels = np.uint8, 3
        elif encoding == 'mono8':
            dtype, n_channels = np.uint8, 1
        elif encoding == 'mono16':
            dtype, n_channels = np.uint16, 1
        elif encoding in ['32FC1', 'passthrough'] and img_msg.encoding == '32FC1':
            dtype, n_channels = np.float32, 1
        else:
            raise ValueError(f"不支持的编码: {encoding}")
        
        # 将字节数据转换为numpy数组
        if n_channels == 1:
            im = np.frombuffer(img_msg.data, dtype=dtype).reshape(
                img_msg.height, img_msg.width)
        else:
            im = np.frombuffer(img_msg.data, dtype=dtype).reshape(
                img_msg.height, img_msg.width, n_channels)
            
        # 颜色空间转换
        if desired_encoding == "bgr8" and img_msg.encoding == "rgb8":
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        elif desired_encoding == "rgb8" and img_msg.encoding == "bgr8":
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
        return im