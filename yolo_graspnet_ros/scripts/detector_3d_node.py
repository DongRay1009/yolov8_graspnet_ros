#!/usr/bin/env python3
import os
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将脚本目录添加到Python路径
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

import rospy
import cv2
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import tf2_ros
import tf.transformations
from ultralytics import YOLO  # 导入ultralytics包来加载YOLOv8模型

class YoloV8Detector:
    def __init__(self, model_path):
        """初始化YOLOv8检测器使用PyTorch模型"""
        try:
            self.model = YOLO(model_path)  # 加载PyTorch模型
            self.confidence_threshold = 0.25
            rospy.loginfo(f"成功加载YOLOv8 PyTorch模型: {model_path}")
        except Exception as e:
            rospy.logerr(f"加载模型失败: {e}")
            raise e
        
    def detect(self, image):
        """使用YOLOv8 PyTorch模型进行目标检测"""
        # 使用ultralytics API预测
        results = self.model(image, verbose=False)
        
        # 获取检测结果
        detections = []
        
        for result in results:
            # 如果有检测到的目标
            if result.boxes is not None and len(result.boxes) > 0:
                # 获取边界框、置信度和类别
                boxes = result.boxes.xyxy.cpu().numpy()    # 以(x1,y1,x2,y2)格式获取边界框
                confs = result.boxes.conf.cpu().numpy()    # 获取置信度
                clss = result.boxes.cls.cpu().numpy()      # 获取类别索引
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confs, clss)):
                    # 过滤低置信度检测
                    if conf < self.confidence_threshold:
                        continue
                        
                    # 获取边界框坐标(左上和右下)
                    x1, y1, x2, y2 = box
                    
                    # 计算中心点和宽高
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # 注释掉详细的检测信息日志
                    # rospy.loginfo(f"检测到物体: 中心点=({center_x},{center_y}), " 
                    #              f"宽高=({width},{height}), 类别={int(cls)}, 置信度={conf:.3f}")
                    
                    detections.append({
                        'x': center_x,
                        'y': center_y,
                        'width': width,
                        'height': height,
                        'class_id': int(cls),
                        'confidence': float(conf)
                    })
            
        return detections

class Detector3DNode:
    def __init__(self):
        rospy.init_node('detector_3d_node_py')
        model_path = rospy.get_param('~model_path', '/home/msi/yolo_3d_ws/src/yolo_graspnet_ros/models/yolov8m.pt')
        rgb_topic = rospy.get_param('~rgb_topic', '/rgb/image_raw')
        depth_topic = rospy.get_param('~depth_topic', '/depth_to_rgb/image_raw')
        camera_info_topic = rospy.get_param('~camera_info', '/rgb/camera_info')
        
        self.bridge = CvBridge()
        self.detector = YoloV8Detector(model_path)
        
        self.rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.image_callback)
        
        self.detection_3d_pub = rospy.Publisher('/object_poses', PoseArray, queue_size=1)
        self.depth_debug_pub = rospy.Publisher('depth_debug', Image, queue_size=1)
        self.debug_pub = rospy.Publisher('detection_debug', Image, queue_size=1)
        
        self.camera_matrix = None
        self.dist_coeffs = None
        
        self.coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                     "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                     "hair drier", "toothbrush"]
        
        rospy.loginfo("YOLOv8 3D Detector Node Initialized")
    
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            try:
                self.camera_matrix = np.array(msg.K).reshape(3, 3)
                self.dist_coeffs = np.array(msg.D)
                rospy.loginfo("相机内参初始化成功")
                # 注释掉内参矩阵详细输出
                # rospy.loginfo(f"相机矩阵: \n{self.camera_matrix}")
            except Exception as e:
                rospy.logerr(f"相机内参初始化失败: {e}")
    
    def get_3d_point_robust(self, x, y, depth_image, window_size=51):
        """使用区域采样获取更稳定的深度值"""
        if self.camera_matrix is None:
            return None
        
        height, width = depth_image.shape
        x = max(0, min(int(x), width-1))
        y = max(0, min(int(y), height-1))
        
        # 使用多个窗口尺寸尝试获取深度
        for size in [window_size, window_size*2, window_size*3]:
            half_size = size // 2
            x_min = max(0, x - half_size)
            x_max = min(width-1, x + half_size)
            y_min = max(0, y - half_size)
            y_max = min(height-1, y + half_size)
            
            # 提取窗口区域
            window = depth_image[y_min:y_max+1, x_min:x_max+1].astype(np.float32)
            
            # 过滤有效深度值（排除0和异常值）
            min_depth = 50   # 最小有效深度值（毫米）
            max_depth = 10000  # 最大有效深度值（毫米）
            valid_depths = window[(window > min_depth) & (window < max_depth)]
            
            # 绘制调试窗口
            depth_viz = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.rectangle(depth_viz, (x_min, y_min), (x_max, y_max), (0,255,255), 2)
            cv2.circle(depth_viz, (x, y), 5, (0,0,255), -1)
            # cv2.imshow("Depth Search Window", depth_viz)
            # cv2.waitKey(1)
            
            if len(valid_depths) >= 20:  # 要求至少有20个有效点
                # 使用中值作为稳健估计
                depth = np.median(valid_depths)
                depth_meters = depth * 0.001  # 毫米转米
                
                # 计算3D坐标
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]
                cx = self.camera_matrix[0, 2]
                cy = self.camera_matrix[1, 2]
                
                x_3d = (x - cx) * depth_meters / fx
                y_3d = (y - cy) * depth_meters / fy
                z_3d = depth_meters
                
                # 注释掉详细的深度点信息
                # rospy.loginfo(f"找到有效深度点: {len(valid_depths)}个, 深度值: {depth_meters:.3f}m")
                # rospy.loginfo(f"计算的3D坐标: X={x_3d:.3f}, Y={y_3d:.3f}, Z={z_3d:.3f}")
                return (x_3d, y_3d, z_3d)
        
        rospy.logwarn(f"位置({x},{y})周围没有足够的有效深度点")
        return None
    
    def visualize_depth_map(self, depth_image):
        """可视化深度图并统计有效点"""
        # 创建深度图可视化
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # 计算有效深度点的比例
        valid_points = np.count_nonzero((depth_image > 100) & (depth_image < 10000))
        total_points = depth_image.size
        valid_ratio = valid_points / total_points * 100
        
        # 添加信息到图像上
        cv2.putText(depth_colormap, f"有效深度点: {valid_ratio:.2f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # cv2.imshow("Depth Map", depth_colormap)
        # cv2.waitKey(1)
        
        # 发布深度调试图像
        try:
            self.depth_debug_pub.publish(self.bridge.cv2_to_imgmsg(depth_colormap, "bgr8"))
        except:
            pass
            
        # 注释掉深度图统计信息
        # rospy.loginfo(f"深度图有效点占比: {valid_ratio:.2f}%, 有效点数: {valid_points}/{total_points}")
        return valid_ratio
        
    def image_callback(self, rgb_msg, depth_msg):
        if self.camera_matrix is None:
            rospy.logerr("相机内参未初始化！")
            return
        
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            # 注释掉图像接收日志
            # rospy.loginfo("接收到RGB和深度图像")
            # rospy.loginfo(f"RGB图像尺寸: {cv_rgb.shape}, 深度图像尺寸: {cv_depth.shape}")
            
            # 可视化深度图
            self.visualize_depth_map(cv_depth)
            
            # 执行物体检测
            detections = self.detector.detect(cv_rgb)
            
            pose_array = PoseArray()
            pose_array.header = rgb_msg.header
            
            detected_objects_count = 0
            last_valid_point = None
            
            for det in detections:
                # 过滤图像边界检测
                if (det['x'] < 10 or det['x'] > cv_rgb.shape[1]-10 or 
                    det['y'] < 10 or det['y'] > cv_rgb.shape[0]-10):
                    continue
                
                class_id = det['class_id']
                conf = det['confidence']
                x, y = int(det['x']), int(det['y'])
                
                # 注释掉检测坐标日志
                # rospy.loginfo(f"检测坐标: ({x}, {y})")
                
                class_name = self.coco_names[class_id] if class_id < len(self.coco_names) else f"类别{class_id}"
                
                # 使用增强的深度估计方法
                point_3d = self.get_3d_point_robust(x, y, cv_depth, window_size=51)
                
                if point_3d is None:
                    continue
                
                last_valid_point = (x, y)
                detected_objects_count += 1
                x_3d, y_3d, z_3d = point_3d
                
                # 保留三维坐标和物体标签的输出
                rospy.loginfo(f"检测到物体: {class_name}, 置信度: {conf:.2f}, 3D坐标: X={x_3d:.3f}, Y={y_3d:.3f}, Z={z_3d:.3f}")
                    
                pose = Pose()
                pose.position.x = x_3d
                pose.position.y = y_3d
                pose.position.z = z_3d
                pose.orientation.w = 1.0
                
                pose_array.poses.append(pose)
                
                # 绘制检测框和3D信息
                cv2.putText(cv_rgb, f"{class_name}: {conf:.2f}", 
                            (int(det['x'] - det['width']/2), int(det['y'] - det['height']/2 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(cv_rgb, f"X:{x_3d:.2f} Y:{y_3d:.2f} Z:{z_3d:.2f}m", 
                            (int(det['x'] - det['width']/2), int(det['y'] - det['height']/2 + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                cv2.rectangle(cv_rgb, 
                            (int(det['x'] - det['width']/2), int(det['y'] - det['height']/2)), 
                            (int(det['x'] + det['width']/2), int(det['y'] + det['height']/2)), 
                            (0, 255, 0), 2)
            
            # 注释掉总结检测数量日志
            # rospy.loginfo(f"检测到 {len(detections)} 个物体，其中 {detected_objects_count} 个有有效3D位置")
            
            # 发布结果
            self.detection_3d_pub.publish(pose_array)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_rgb, "bgr8"))
            
            cv2.imshow("YOLOv8 3D Detections", cv_rgb)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    try:
        node = Detector3DNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()