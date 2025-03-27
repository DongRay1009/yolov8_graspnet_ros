#!/home/msi/python_envs/yolo_env/bin/python3
# filepath: /home/msi/yolo_3d_ws/src/yolo_graspnet_ros/scripts/graspnet_generator_with_dl.py

import rospy
import numpy as np
import cv2
import torch
import os
import sys
import traceback
import tf2_ros
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from visualization_msgs.msg import MarkerArray
from std_srvs.srv import Empty

# 导入自定义模块
from utils.bridge_utils import get_bridge
from utils.pointcloud_utils import prepare_pointcloud_for_graspnet
from utils.dl_model_utils import load_graspnet_model, create_camera_info, predict_grasps
from utils.grasp_utils import create_default_grasp
from utils.visualization import create_grasp_marker

class GraspNetROS:
    def __init__(self):
        """初始化GraspNet ROS节点"""
        rospy.init_node('graspnet_ros')
        
        # 获取参数
        self.rgb_topic = rospy.get_param('~rgb_topic', '/rgb/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/depth_to_rgb/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info', '/rgb/camera_info')
        self.detection_topic = rospy.get_param('~detection_topic', '/object_poses')
        
        # 设置GraspNet模型路径
        self.model_dir = rospy.get_param('~model_dir', 
                                        '/home/msi/yolo_3d_ws/src/yolo_graspnet_ros/models/graspnet')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint.tar')
        
        rospy.loginfo(f"使用模型路径: {self.checkpoint_path}")
        
        # 初始化转换桥
        self.bridge = get_bridge()
        
        # 设置发布者
        self.grasp_pub = rospy.Publisher('/best_grasp_pose', PoseStamped, queue_size=1)
        self.all_grasps_pub = rospy.Publisher('/all_grasp_poses', PoseArray, queue_size=1)
        self.debug_pub = rospy.Publisher('/grasp_debug', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/grasp_markers', MarkerArray, queue_size=1)
        self.grasp_cloud_pub = rospy.Publisher('/grasp_cloud', PointCloud2, queue_size=1)
        
        # 替换为单独的消息订阅
        self.rgb_msg = None
        self.depth_msg = None
        self.poses_msg = None
        
        # 单独订阅
        self.rgb_sub = rospy.Subscriber(
            self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self.depth_callback, queue_size=1)
        self.detection_sub = rospy.Subscriber(
            self.detection_topic, PoseArray, self.detection_callback, queue_size=1)
        
        # 添加定时器进行处理
        self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_callback)
        
        # 获取相机内参
        self.camera_info = None
        self.camera_matrix = None
        self.get_camera_info()
        
        # 初始化TF监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 添加是否强制使用简化版实现的参数
        self.force_simplified = rospy.get_param('~force_simplified', False)
        
        # 修改模型加载代码，传递force_simplified参数
        if self.force_simplified:
            rospy.loginfo("配置为强制使用简化版GraspNet实现")
        
        # 加载GraspNet模型
        self.net, self.pred_decode, self.device = load_graspnet_model(self.checkpoint_path)
        self.model_loaded = self.net is not None
        
        rospy.loginfo("GraspNet ROS节点初始化完成")
    
    def get_camera_info(self):
        """获取相机内参"""
        try:
            self.camera_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo, timeout=5.0)
            K = self.camera_info.K
            self.camera_matrix = np.array([[K[0], K[1], K[2]], 
                                          [K[3], K[4], K[5]], 
                                          [K[6], K[7], K[8]]])
            rospy.loginfo("相机内参初始化成功")
            rospy.loginfo(f"相机矩阵: \n{self.camera_matrix}")
        except rospy.ROSException:
            rospy.logerr("无法获取相机内参")
            self.camera_matrix = np.array([[550.0, 0, 320.0], [0, 550.0, 240.0], [0, 0, 1]])
    
    def rgb_callback(self, msg):
        """RGB图像回调函数"""
        self.rgb_msg = msg
        rospy.loginfo_throttle(2.0, "收到RGB图像")
    
    def depth_callback(self, msg):
        """深度图像回调函数"""
        self.depth_msg = msg
        rospy.loginfo_throttle(2.0, "收到深度图像")
    
    def detection_callback(self, msg):
        """物体检测结果回调函数"""
        self.poses_msg = msg
        rospy.loginfo_throttle(2.0, f"收到检测结果, 包含 {len(msg.poses)} 个物体")
    
    def timer_callback(self, event):
        """定时器回调函数，处理收集到的消息"""
        # 先确保模型已加载
        if not hasattr(self, 'model_loaded') or self.model_loaded is None:
            rospy.loginfo_throttle(2.0, "模型尚未加载完成，跳过处理")
            return
        
        # 检查是否所有必要消息都已收到
        if self.rgb_msg is None or self.depth_msg is None:
            return
            
        # 没有检测结果时也可以尝试生成抓取
        if self.poses_msg is None:
            rospy.loginfo_throttle(2.0, "没有收到物体检测结果，使用全场景")
            
        # 复制消息以避免处理时被其他回调修改
        rgb_msg = self.rgb_msg
        depth_msg = self.depth_msg
        poses_msg = self.poses_msg
        
        # 重置消息，以便下一次处理新消息
        self.rgb_msg = None
        self.depth_msg = None
        self.poses_msg = None
        
        # 处理消息
        self.process_messages(rgb_msg, depth_msg, poses_msg)
    
    def process_messages(self, rgb_msg, depth_msg, poses_msg=None):
        """处理收集到的RGB、深度和检测结果消息"""
        try:
            rospy.loginfo("处理新的消息")
            
            # 检查poses_msg是否为None
            if poses_msg is None:
                rospy.loginfo("没有检测结果，将使用全场景生成抓取位姿")
                pose_count = 0
            else:
                pose_count = len(poses_msg.poses)
                # rospy.loginfo(f"检测到的目标数量: {pose_count}")
            
            # 转换图像
            try:
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
                depth = self.bridge.imgmsg_to_cv2(depth_msg)
                # rospy.loginfo(f"图像转换成功: RGB {rgb.shape}, 深度 {depth.shape}")
            except Exception as e:
                rospy.logerr(f"图像转换失败: {e}")
                return
                
            # 配置相机信息
            camera_info = create_camera_info(
                self.camera_matrix[0, 0],  # fx
                self.camera_matrix[1, 1],  # fy
                self.camera_matrix[0, 2],  # cx
                self.camera_matrix[1, 2],  # cy
                rgb.shape[1],              # width
                rgb.shape[0]               # height
            )
            
            # 根据是否有检测结果采取不同处理
            if poses_msg is None or pose_count == 0:
                # 没有检测结果，使用全场景生成抓取位姿
                if self.model_loaded:
                    # 直接使用已加载的模型，不重新导入
                    grasp_results = predict_grasps(self.net, self.pred_decode, self.device, rgb, depth, camera_info)
                    if grasp_results and len(grasp_results) > 0:
                        rospy.loginfo(f"全场景生成了 {len(grasp_results)} 个抓取候选")
                        self.process_grasp_results(grasp_results)
                    else:
                        rospy.logwarn("没有生成有效的抓取候选")
                        self.publish_default_grasp()
                else:
                    rospy.logwarn("模型未加载，无法生成抓取位姿")
                    self.publish_default_grasp()
            else:
                # 有检测结果，为每个物体生成抓取位姿
                for i, pose in enumerate(poses_msg.poses):
                    position = pose.position
                    rospy.loginfo(f"为物体 {i+1} 生成抓取位姿，位置: ({position.x:.3f}, {position.y:.3f}, {position.z:.3f})")
                    
                    # 生成抓取位姿
                    if self.model_loaded:
                        grasp_results = predict_grasps(self.net, self.pred_decode, self.device, rgb, depth, camera_info)
                        
                        if grasp_results and len(grasp_results) > 0:
                            rospy.loginfo(f"为物体 {i+1} 生成了 {len(grasp_results)} 个抓取候选")
                            self.process_grasp_results(grasp_results)
                        else:
                            rospy.logwarn(f"为物体 {i+1} 没有生成有效的抓取候选")
                            
                    # 只处理第一个物体
                    break
        
        except Exception as e:
            rospy.logerr(f"消息处理异常: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            
    # 添加新方法处理抓取结果
    def process_grasp_results(self, grasp_results):
        """处理抓取结果并发布"""
        try:
            # 过滤有效抓取位姿
            valid_grasps = self.filter_valid_grasps(grasp_results)
            
            # 发布最佳抓取位姿
            best_grasp = valid_grasps[0]  # 结果已经按分数排序
            best_pose = PoseStamped()
            best_pose.header.frame_id = "camera_color_optical_frame"
            best_pose.header.stamp = rospy.Time.now()
            
            # 设置位置（转换为米）
            best_pose.pose.position.x = float(best_grasp['point'][0]) / 1000.0
            best_pose.pose.position.y = float(best_grasp['point'][1]) / 1000.0
            best_pose.pose.position.z = float(best_grasp['point'][2]) / 1000.0
            
            # 设置方向
            try:
                import tf.transformations
                matrix = np.eye(4)
                matrix[:3, :3] = best_grasp['rotation']
                q = tf.transformations.quaternion_from_matrix(matrix)
                best_pose.pose.orientation.x = q[0]
                best_pose.pose.orientation.y = q[1]
                best_pose.pose.orientation.z = q[2]
                best_pose.pose.orientation.w = q[3]
            except:
                # 转换失败时使用默认方向
                best_pose.pose.orientation.w = 1.0
            
            # 发布最佳抓取位姿
            self.grasp_pub.publish(best_pose)
            rospy.loginfo(f"发布最佳抓取位姿: " +
                          f"位置=({best_pose.pose.position.x:.3f}, {best_pose.pose.position.y:.3f}, {best_pose.pose.position.z:.3f}), " +
                          f"分数={best_grasp['score']:.3f}")
            
            # 创建并发布标记数组
            markers = MarkerArray()
            for j, grasp in enumerate(valid_grasps[:10]):  # 只显示前10个
                marker = create_grasp_marker(
                    grasp['point'],
                    grasp['rotation'],
                    grasp['width'],
                    grasp['score'],
                    j
                )
                markers.markers.append(marker)
            
            if len(markers.markers) > 0:
                self.marker_pub.publish(markers)
                # rospy.loginfo(f"发布了 {len(markers.markers)} 个抓取标记")
                
            # 发布抓取点云
            self.publish_grasp_cloud(grasp_results)
                
        except Exception as e:
            rospy.logerr(f"处理抓取结果失败: {e}")
            self.publish_default_grasp()
    
    def filter_valid_grasps(self, grasp_results):
        """过滤出有效的抓取位姿"""
        valid_grasps = []
        for grasp in grasp_results:
            point = grasp['point']
            # 示例：过滤掉过远或过近的点
            distance = np.linalg.norm(point)
            if 0.05 < distance < 1.5:  # 单位：米
                valid_grasps.append(grasp)
        
        return valid_grasps if valid_grasps else grasp_results  # 如果全都过滤掉了，返回原始结果
    
    def create_pose_stamped(self, pose):
        """从Pose创建PoseStamped消息"""
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "camera_color_optical_frame"
        pose_stamped.pose = pose
        return pose_stamped
    
    def predict_simple_grasp(self, rgb_roi, depth_roi, cx, cy):
        """简单的抓取位姿预测（当GraspNet不可用时）"""
        import tf.transformations
        
        depth_values = depth_roi[depth_roi > 0.01]  # 过滤无效值
        if len(depth_values) == 0:
            z = 0.5  # 默认深度
        else:
            z = np.median(depth_values)  # 使用中值深度
        
        # 将图像坐标转换为相机坐标
        x = (cx - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
        y = (cy - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
        
        # 创建位姿消息
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "camera_color_optical_frame"
        
        # 设置位置（相机坐标系中的位置）
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z - 0.05  # 略微靠近物体
        
        # 设置方向（抓取方向朝下）
        q = tf.transformations.quaternion_from_euler(-np.pi/2, 0, 0)
        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]
        
        # 计算一个简单的评分
        score = 0.7  # 示例评分
        
        return pose_stamped, score
    
    def publish_default_grasp(self):
        """发布默认抓取位姿"""
        default_pose = create_default_grasp()
        self.grasp_pub.publish(default_pose)
        rospy.loginfo("发布默认抓取位姿")
    
    def rotation_matrix_to_quaternion(self, rot_matrix):
        """将旋转矩阵转换为四元数"""
        import tensorflow as tf
        
        # 确保输入是有效的旋转矩阵
        if isinstance(rot_matrix, np.ndarray):
            if rot_matrix.shape != (3, 3):
                rospy.logwarn(f"旋转矩阵形状不正确: {rot_matrix.shape}")
                # 如果不是3x3矩阵，返回默认方向（朝下）
                return [0, 0, -0.707, 0.707]
        
        try:
            # 计算四元数的元素
            trace = np.trace(rot_matrix)
            
            if trace > 0:
                S = np.sqrt(trace + 1.0) * 2
                qw = 0.25 * S
                qx = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
                qy = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
                qz = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
            elif rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
                S = np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
                qw = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
                qx = 0.25 * S
                qy = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
                qz = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
            elif rot_matrix[1, 1] > rot_matrix[2, 2]:
                S = np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2
                qw = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
                qx = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
                qy = 0.25 * S
                qz = (rot_matrix[1, 2] + rot_matrix[2, 1]) / S
            else:
                S = np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2
                qw = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
                qx = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
                qy = (rot_matrix[1, 2] + rot_matrix[2, 1]) / S
                qz = 0.25 * S
            
            return [qx, qy, qz, qw]
        except Exception as e:
            rospy.logerr(f"四元数转换错误: {e}")
            # 返回默认方向（朝下）
            return [0, 0, -0.707, 0.707]
    
    # 在发布抓取标记的同时发布点云
    def publish_grasp_cloud(self, grasp_results):
        """发布抓取点云用于可视化"""
        from sensor_msgs.msg import PointCloud2, PointField
        import sensor_msgs.point_cloud2 as pc2
        
        if not grasp_results:
            return
            
        # 创建点云消息
        cloud_msg = PointCloud2()
        cloud_msg.header.frame_id = "camera_color_optical_frame"
        cloud_msg.header.stamp = rospy.Time.now()
        
        # 提取所有抓取点
        points = []
        for grasp in grasp_results:
            # 转换为米制单位
            x, y, z = grasp['point'] / 1000.0
            r, g, b = int(255 * (1.0 - grasp['score'])), int(255 * grasp['score']), 0
            points.append([x, y, z, r, g, b])
        
        # 创建点云字段
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.UINT8, count=1),
            PointField(name='g', offset=13, datatype=PointField.UINT8, count=1),
            PointField(name='b', offset=14, datatype=PointField.UINT8, count=1),
        ]
        
        # 创建点云
        cloud_msg = pc2.create_cloud(cloud_msg.header, fields, points)
        self.grasp_cloud_pub.publish(cloud_msg)

if __name__ == "__main__":
    try:
        graspnet_node = GraspNetROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass