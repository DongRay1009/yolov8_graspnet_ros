#!/usr/bin/env python3
import os
import sys

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add script directory to Python path
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
from ultralytics import YOLO  # Import ultralytics package to load YOLOv8 model

class YoloV8Detector:
    def __init__(self, model_path):
        """Initialize YOLOv8 detector using PyTorch model"""
        try:
            self.model = YOLO(model_path)  # Load PyTorch model
            # MODIFY: Adjust confidence threshold based on your application needs
            self.confidence_threshold = 0.25
            rospy.loginfo(f"Successfully loaded YOLOv8 PyTorch model: {model_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            raise e
        
    def detect(self, image):
        """Perform object detection using YOLOv8 PyTorch model"""
        # Use ultralytics API for prediction
        results = self.model(image, verbose=False)
        
        # Get detection results
        detections = []
        
        for result in results:
            # If objects are detected
            if result.boxes is not None and len(result.boxes) > 0:
                # Get bounding boxes, confidence and class
                boxes = result.boxes.xyxy.cpu().numpy()    # Get bounding boxes in (x1,y1,x2,y2) format
                confs = result.boxes.conf.cpu().numpy()    # Get confidence values
                clss = result.boxes.cls.cpu().numpy()      # Get class indices
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confs, clss)):
                    # Filter low confidence detections
                    if conf < self.confidence_threshold:
                        continue
                        
                    # Get bounding box coordinates (top-left and bottom-right)
                    x1, y1, x2, y2 = box
                    
                    # Calculate center point and dimensions
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # Commented out detailed detection info log
                    # rospy.loginfo(f"Detected object: center=({center_x},{center_y}), " 
                    #              f"dimensions=({width},{height}), class={int(cls)}, confidence={conf:.3f}")
                    
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
        # MODIFY: Change these paths to match your setup
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
        
        # COCO dataset class names
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
                rospy.loginfo("Camera intrinsics initialization successful")
                # Commented out detailed camera matrix output
                # rospy.loginfo(f"Camera matrix: \n{self.camera_matrix}")
            except Exception as e:
                rospy.logerr(f"Camera intrinsics initialization failed: {e}")
    
    def get_3d_point_robust(self, x, y, depth_image, window_size=51):
        """Get more stable depth values using area sampling"""
        if self.camera_matrix is None:
            return None
        
        height, width = depth_image.shape
        x = max(0, min(int(x), width-1))
        y = max(0, min(int(y), height-1))
        
        # Try multiple window sizes to get depth
        for size in [window_size, window_size*2, window_size*3]:
            half_size = size // 2
            x_min = max(0, x - half_size)
            x_max = min(width-1, x + half_size)
            y_min = max(0, y - half_size)
            y_max = min(height-1, y + half_size)
            
            # Extract window region
            window = depth_image[y_min:y_max+1, x_min:x_max+1].astype(np.float32)
            
            # Filter valid depth values (exclude 0 and abnormal values)
            # MODIFY: Adjust min_depth and max_depth based on your camera and scene
            min_depth = 50   # Minimum valid depth value (mm)
            max_depth = 10000  # Maximum valid depth value (mm)
            valid_depths = window[(window > min_depth) & (window < max_depth)]
            
            # Draw debug window
            depth_viz = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.rectangle(depth_viz, (x_min, y_min), (x_max, y_max), (0,255,255), 2)
            cv2.circle(depth_viz, (x, y), 5, (0,0,255), -1)
            # cv2.imshow("Depth Search Window", depth_viz)
            # cv2.waitKey(1)
            
            if len(valid_depths) >= 20:  # Require at least 20 valid points
                # Use median as robust estimate
                depth = np.median(valid_depths)
                depth_meters = depth * 0.001  # Convert mm to meters
                
                # Calculate 3D coordinates
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]
                cx = self.camera_matrix[0, 2]
                cy = self.camera_matrix[1, 2]
                
                x_3d = (x - cx) * depth_meters / fx
                y_3d = (y - cy) * depth_meters / fy
                z_3d = depth_meters
                
                # Commented out detailed depth point info
                # rospy.loginfo(f"Found valid depth points: {len(valid_depths)}, depth value: {depth_meters:.3f}m")
                # rospy.loginfo(f"Calculated 3D coordinates: X={x_3d:.3f}, Y={y_3d:.3f}, Z={z_3d:.3f}")
                return (x_3d, y_3d, z_3d)
        
        rospy.logwarn(f"Not enough valid depth points around position ({x},{y})")
        return None
    
    def visualize_depth_map(self, depth_image):
        """Visualize depth map and count valid points"""
        # Create depth map visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # Calculate proportion of valid depth points
        valid_points = np.count_nonzero((depth_image > 100) & (depth_image < 10000))
        total_points = depth_image.size
        valid_ratio = valid_points / total_points * 100
        
        # Add information to the image
        cv2.putText(depth_colormap, f"Valid depth points: {valid_ratio:.2f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # cv2.imshow("Depth Map", depth_colormap)
        # cv2.waitKey(1)
        
        # Publish depth debug image
        try:
            self.depth_debug_pub.publish(self.bridge.cv2_to_imgmsg(depth_colormap, "bgr8"))
        except:
            pass
            
        # Commented out depth map statistics info
        # rospy.loginfo(f"Depth map valid points: {valid_ratio:.2f}%, valid count: {valid_points}/{total_points}")
        return valid_ratio
        
    def image_callback(self, rgb_msg, depth_msg):
        if self.camera_matrix is None:
            rospy.logerr("Camera intrinsics not initialized!")
            return
        
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            # Commented out image reception log
            # rospy.loginfo("Received RGB and depth images")
            # rospy.loginfo(f"RGB image size: {cv_rgb.shape}, depth image size: {cv_depth.shape}")
            
            # Visualize depth map
            self.visualize_depth_map(cv_depth)
            
            # Perform object detection
            detections = self.detector.detect(cv_rgb)
            
            pose_array = PoseArray()
            pose_array.header = rgb_msg.header
            
            detected_objects_count = 0
            last_valid_point = None
            
            for det in detections:
                # Filter detections at image boundaries
                if (det['x'] < 10 or det['x'] > cv_rgb.shape[1]-10 or 
                    det['y'] < 10 or det['y'] > cv_rgb.shape[0]-10):
                    continue
                
                class_id = det['class_id']
                conf = det['confidence']
                x, y = int(det['x']), int(det['y'])
                
                # Commented out detection coordinates log
                # rospy.loginfo(f"Detection coordinates: ({x}, {y})")
                
                class_name = self.coco_names[class_id] if class_id < len(self.coco_names) else f"Class{class_id}"
                
                # Use enhanced depth estimation method
                point_3d = self.get_3d_point_robust(x, y, cv_depth, window_size=51)
                
                if point_3d is None:
                    continue
                
                last_valid_point = (x, y)
                detected_objects_count += 1
                x_3d, y_3d, z_3d = point_3d
                
                # Keep 3D coordinate and object label output
                rospy.loginfo(f"Detected object: {class_name}, confidence: {conf:.2f}, 3D coordinates: X={x_3d:.3f}, Y={y_3d:.3f}, Z={z_3d:.3f}")
                    
                pose = Pose()
                pose.position.x = x_3d
                pose.position.y = y_3d
                pose.position.z = z_3d
                pose.orientation.w = 1.0
                
                pose_array.poses.append(pose)
                
                # Draw detection box and 3D information
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
            
            # Commented out detection summary log
            # rospy.loginfo(f"Detected {len(detections)} objects, {detected_objects_count} with valid 3D positions")
            
            # Publish results
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