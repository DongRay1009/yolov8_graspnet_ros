#!/usr/bin/env python3
import os
import sys

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add utils directory to Python path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

# Import other modules after setup
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
from visualization_msgs.msg import Marker, MarkerArray 
from std_srvs.srv import Empty
from std_msgs.msg import String

# Import custom modules
from utils.bridge_utils import get_bridge
from utils.pointcloud_utils import prepare_pointcloud_for_graspnet
from utils.dl_model_utils import load_graspnet_model, create_camera_info, predict_grasps
from utils.grasp_utils import create_default_grasp
from utils.visualization import create_grasp_marker

# Import custom messages
from yolo_graspnet_ros.msg import DetectedObject, DetectedObjectArray

class GraspNetROS:
    def __init__(self):
        """Initialize GraspNet ROS node"""
        rospy.init_node('graspnet_ros')
        
        # Get parameters
        self.rgb_topic = rospy.get_param('~rgb_topic', '/rgb/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/depth_to_rgb/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info', '/rgb/camera_info')
        self.detection_topic = rospy.get_param('~detection_topic', '/object_poses')
        
        # Set GraspNet model path
        self.model_dir = rospy.get_param('~model_dir', 
                                        '/home/msi/yolo_3d_ws/src/yolo_graspnet_ros/models/graspnet')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint.tar')
        
        rospy.loginfo(f"Using model path: {self.checkpoint_path}")
        
        # Initialize conversion bridge
        self.bridge = get_bridge()
        
        # Setup publishers
        self.grasp_pub = rospy.Publisher('/best_grasp_pose', PoseStamped, queue_size=1)
        self.all_grasps_pub = rospy.Publisher('/all_grasp_poses', PoseArray, queue_size=1)
        self.debug_pub = rospy.Publisher('/grasp_debug', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/grasp_markers', MarkerArray, queue_size=1)
        self.grasp_cloud_pub = rospy.Publisher('/grasp_cloud', PointCloud2, queue_size=1)
        self.grasp_info_pub = rospy.Publisher('/grasp_info', String, queue_size=10)  # Add information publisher
        
        # Replace with individual message subscriptions
        self.rgb_msg = None
        self.depth_msg = None
        self.poses_msg = None
        
        # Individual subscriptions
        self.rgb_sub = rospy.Subscriber(
            self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self.depth_callback, queue_size=1)
        self.detection_sub = rospy.Subscriber(
            self.detection_topic, DetectedObjectArray, self.detection_callback, queue_size=1)
        
        # COCO class names - for label mapping
        self.coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                     "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                     "hair drier", "toothbrush"]
        
        # Maintain a dictionary of latest detections for label mapping
        self.latest_detections = {}
        
        # Add timer for processing
        self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_callback)
        
        # Get camera intrinsics
        self.camera_info = None
        self.camera_matrix = None
        self.get_camera_info()
        
        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Add parameter to force simplified implementation
        self.force_simplified = rospy.get_param('~force_simplified', False)
        
        # Modify model loading code, pass force_simplified parameter
        if self.force_simplified:
            rospy.loginfo("Configured to force simplified GraspNet implementation")
        else:
            rospy.loginfo("Attempting to load full GraspNet model")
        
        # Load GraspNet model - pass force_simplified parameter
        self.net, self.pred_decode, self.device = load_graspnet_model(
            self.checkpoint_path, 
            force_simplified=self.force_simplified
        )
        self.model_loaded = self.net is not None
        
        rospy.loginfo("GraspNet ROS node initialization complete")
    
    def get_camera_info(self):
        """Get camera intrinsics"""
        try:
            self.camera_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo, timeout=5.0)
            K = self.camera_info.K
            self.camera_matrix = np.array([[K[0], K[1], K[2]], 
                                          [K[3], K[4], K[5]], 
                                          [K[6], K[7], K[8]]])
            rospy.loginfo("Camera intrinsics initialization successful")
            rospy.loginfo(f"Camera matrix: \n{self.camera_matrix}")
        except rospy.ROSException:
            rospy.logerr("Unable to get camera intrinsics")
            self.camera_matrix = np.array([[550.0, 0, 320.0], [0, 550.0, 240.0], [0, 0, 1]])
    
    def rgb_callback(self, msg):
        """RGB image callback function"""
        self.rgb_msg = msg
        rospy.loginfo_throttle(2.0, "Received RGB image")
    
    def depth_callback(self, msg):
        """Depth image callback function"""
        self.depth_msg = msg
        rospy.loginfo_throttle(2.0, "Received depth image")
    
    def detection_callback(self, msg):
        """Object detection results callback function - now receives DetectedObjectArray"""
        self.poses_msg = msg
        rospy.loginfo_throttle(2.0, f"Received detection results with {len(msg.objects)} objects")
        
        # Directly update detected object label information
        self.update_latest_detections(msg)
    
    def update_latest_detections(self, poses_msg):
        """Update label information based on latest detections"""
        try:
            rospy.loginfo(f"Received detection results with {len(poses_msg.objects)} objects")
            self.poses_msg = poses_msg  # Save latest detection results
            
            # Update object labels - add hasattr check
            for obj in poses_msg.objects:
                label_info = f"Getting label from DetectedObject: {obj.label}"
                # Check if confidence attribute exists
                if hasattr(obj, 'confidence'):
                    label_info += f", confidence: {obj.confidence:.2f}"
                rospy.loginfo(label_info)
        except Exception as e:
            rospy.logerr(f"Error processing detection results: {e}")
    
    def get_object_label(self, position):
        """Try to get object label"""
        # Create position key
        pos_key = f"{position.x:.2f}_{position.y:.2f}_{position.z:.2f}"
        
        # Check if there's a matching label
        for key, label in self.latest_detections.items():
            # Allow some position error
            parts_key = key.split('_')
            parts_pos = pos_key.split('_')
            
            if len(parts_key) == 3 and len(parts_pos) == 3:
                try:
                    key_x, key_y, key_z = float(parts_key[0]), float(parts_key[1]), float(parts_key[2])
                    pos_x, pos_y, pos_z = float(parts_pos[0]), float(parts_pos[1]), float(parts_pos[2])
                    
                    # If position difference is less than threshold, consider it the same object
                    if (abs(key_x - pos_x) < 0.1 and 
                        abs(key_y - pos_y) < 0.1 and 
                        abs(key_z - pos_z) < 0.1):
                        return label
                except:
                    pass
        
        # If no matching label found, return default
        return "Unknown object"
    
    def timer_callback(self, event):
        """Timer callback function to process collected messages"""
        # First ensure the model is loaded
        if not hasattr(self, 'model_loaded') or self.model_loaded is None:
            rospy.loginfo_throttle(2.0, "Model not yet loaded, skipping processing")
            return
        
        # Check if all necessary messages have been received
        if self.rgb_msg is None or self.depth_msg is None:
            return
            
        # Can try to generate grasp even without detection results
        if self.poses_msg is None:
            rospy.loginfo_throttle(2.0, "No object detection results received, using entire scene")
            
        # Copy messages to avoid modification during processing
        rgb_msg = self.rgb_msg
        depth_msg = self.depth_msg
        poses_msg = self.poses_msg
        
        # Reset messages to process new ones next time
        self.rgb_msg = None
        self.depth_msg = None
        self.poses_msg = None
        
        # Process messages
        self.process_messages(rgb_msg, depth_msg, poses_msg)
    
    def process_messages(self, rgb_msg, depth_msg, poses_msg=None):
        """Process collected RGB, depth and detection result messages"""
        try:
            rospy.loginfo("Processing new messages")
            
            # Check if poses_msg is None
            if poses_msg is None:
                rospy.loginfo("No detection results, skipping grasp pose generation")
                return  # Return directly, don't generate grasp poses
            else:
                pose_count = len(poses_msg.objects)
                if pose_count == 0:
                    rospy.loginfo("No objects detected, skipping grasp pose generation")
                    return  # Also return if 0 objects detected
            
            # Convert images
            try:
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
                depth = self.bridge.imgmsg_to_cv2(depth_msg)
            except Exception as e:
                rospy.logerr(f"Image conversion failed: {e}")
                return
                
            # Configure camera info
            camera_info = create_camera_info(
                self.camera_matrix[0, 0],  # fx
                self.camera_matrix[1, 1],  # fy
                self.camera_matrix[0, 2],  # cx
                self.camera_matrix[1, 2],  # cy
                rgb.shape[1],              # width
                rgb.shape[0]               # height
            )
            
            # Different processing based on detection results
            if poses_msg is None or pose_count == 0:
                # No detection results, generate grasp poses for entire scene
                if self.model_loaded:
                    # Use already loaded model, don't reimport
                    grasp_results = predict_grasps(self.net, self.pred_decode, self.device, rgb, depth, camera_info)
                    if grasp_results and len(grasp_results) > 0:
                        rospy.loginfo(f"Generated {len(grasp_results)} grasp candidates for entire scene")
                        self.process_grasp_results(grasp_results)
                    else:
                        rospy.logwarn("No valid grasp candidates generated")
                        self.publish_default_grasp()
                else:
                    rospy.logwarn("Model not loaded, cannot generate grasp poses")
                    self.publish_default_grasp()
            else:
                # Have detection results, generate grasp poses for each object
                self.process_detected_objects(rgb, depth, poses_msg, rgb_msg.header, camera_info)
        
        except Exception as e:
            rospy.logerr(f"Message processing exception: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            
    def process_detected_objects(self, rgb, depth, poses_msg, header, camera_info):
        """Process detected objects"""
        all_grasps = []
        
        # Process each detected object
        for i, obj in enumerate(poses_msg.objects):
            # Get position
            position = obj.pose.position
            # Get label, default to empty string
            label = getattr(obj, 'label', '')
            
            rospy.loginfo(f"Generating grasp poses for object {i+1} ({label}), position: ({position.x:.3f}, {position.y:.3f}, {position.z:.3f})")
            
            # Use GraspNet to generate grasps
            grasp_results = predict_grasps(self.net, self.pred_decode, self.device,
                                          rgb, depth, camera_info, object_position=position)
            
            # Add object information to grasp results, using safe attribute access
            if grasp_results and len(grasp_results) > 0:
                for grasp in grasp_results:
                    grasp['label'] = label
                    # Safely get optional attributes
                    if hasattr(obj, 'class_id'):
                        grasp['class_id'] = obj.class_id
                    # Don't try to access confidence attribute
                    
                rospy.loginfo(f"Generated {len(grasp_results)} grasp candidates for object {i+1} ({label})")
                all_grasps.extend(grasp_results)
            else:
                rospy.logwarn(f"No valid grasp candidates generated for object {i+1} ({label})")
        
        # Process all collected grasps
        if all_grasps:
            self.process_grasp_results(all_grasps, header)
        else:
            rospy.logwarn("No valid grasps generated for any object")
            self.publish_default_grasp(header)
    
    def process_grasp_results(self, grasp_results, header=None):
        """Process grasp results and publish"""
        try:
            # Filter valid grasp poses
            valid_grasps = self.filter_valid_grasps(grasp_results)
            
            if not valid_grasps:
                rospy.logwarn("No valid grasp poses found")
                self.publish_default_grasp(header)
                return
                    
            # Publish best grasp pose
            best_grasp = valid_grasps[0]  # Results already sorted by score
            
            # Create PoseStamped message
            pose_stamped = PoseStamped()
            
            # Use provided header or create new one
            if header:
                pose_stamped.header = header
            else:
                pose_stamped.header.stamp = rospy.Time.now()
                pose_stamped.header.frame_id = "camera_color_optical_frame"
            
            # Set position (meters)
            pose_stamped.pose.position.x = best_grasp['point'][0] / 1000.0
            pose_stamped.pose.position.y = best_grasp['point'][1] / 1000.0
            pose_stamped.pose.position.z = best_grasp['point'][2] / 1000.0
            
            try:
                # Set orientation (convert rotation matrix to quaternion)
                q = self.rotation_matrix_to_quaternion(best_grasp['rotation'])
                pose_stamped.pose.orientation.x = q[0]
                pose_stamped.pose.orientation.y = q[1]
                pose_stamped.pose.orientation.z = q[2]
                pose_stamped.pose.orientation.w = q[3]
            except Exception as e:
                rospy.logerr(f"Orientation conversion error: {e}")
                # Use default orientation (downward)
                pose_stamped.pose.orientation.x = 0
                pose_stamped.pose.orientation.y = 0
                pose_stamped.pose.orientation.z = -0.707
                pose_stamped.pose.orientation.w = 0.707
            
            # Create text marker, display object label and score
            text_marker = Marker()
            text_marker.header = pose_stamped.header
            text_marker.ns = "grasp_labels"
            text_marker.id = 0
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose = pose_stamped.pose
            text_marker.pose.position.z += 0.05  # Display above grasp point
            score_text = f"{best_grasp['score']:.2f}"
            
            # If there's a label, add it to the text
            if 'label' in best_grasp and best_grasp['label']:
                score_text = f"{best_grasp['label']}: {score_text}"
                
            text_marker.text = score_text
            text_marker.scale.z = 0.02  # Text size
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            
            # Create MarkerArray and add text marker
            marker_array = MarkerArray()
            marker_array.markers.append(text_marker)
            
            # Publish grasp pose and markers
            self.grasp_pub.publish(pose_stamped)
            self.marker_pub.publish(marker_array)
            
            # Create string with all information
            label_info = best_grasp.get('label', 'Unknown object')
            grasp_info = f"Object class: {label_info}\n" + \
                         f"Position: ({pose_stamped.pose.position.x:.3f}, {pose_stamped.pose.position.y:.3f}, {pose_stamped.pose.position.z:.3f})\n" + \
                         f"Orientation: quaternion({pose_stamped.pose.orientation.x:.3f}, {pose_stamped.pose.orientation.y:.3f}, " + \
                         f"{pose_stamped.pose.orientation.z:.3f}, {pose_stamped.pose.orientation.w:.3f})\n" + \
                         f"Grasp score: {best_grasp['score']:.3f}"
            
            # Publish complete information
            self.grasp_info_pub.publish(grasp_info)
            
            # Publish point cloud
            self.publish_grasp_cloud(valid_grasps)
            
            label_info = best_grasp.get('label', 'Unknown object')
            rospy.loginfo(f"Published grasp pose: object={label_info}, " +
                        f"position=({pose_stamped.pose.position.x:.3f}, {pose_stamped.pose.position.y:.3f}, {pose_stamped.pose.position.z:.3f}), " +
                        f"orientation=({pose_stamped.pose.orientation.x:.3f}, {pose_stamped.pose.orientation.y:.3f}, " +
                        f"{pose_stamped.pose.orientation.z:.3f}, {pose_stamped.pose.orientation.w:.3f}), " +
                        f"score={best_grasp['score']:.2f}")
        
        except Exception as e:
            rospy.logerr(f"Failed to process grasp results: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            self.publish_default_grasp(header)
    
    def filter_valid_grasps(self, grasp_results):
        """Filter valid grasp poses"""
        valid_grasps = []
        for grasp in grasp_results:
            point = grasp['point']
            # Example: filter out points too far or too close
            distance = np.linalg.norm(point)
            if 0.05 < distance < 1.5:  # Units: meters
                valid_grasps.append(grasp)
        
        return valid_grasps if valid_grasps else grasp_results  # If all filtered out, return original results
    
    def create_pose_stamped(self, pose):
        """Create PoseStamped message from Pose"""
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "camera_color_optical_frame"
        pose_stamped.pose = pose
        return pose_stamped
    
    def predict_simple_grasp(self, rgb_roi, depth_roi, cx, cy):
        """Simple grasp pose prediction (when GraspNet is unavailable)"""
        import tf.transformations
        
        depth_values = depth_roi[depth_roi > 0.01]  # Filter invalid values
        if len(depth_values) == 0:
            z = 0.5  # Default depth
        else:
            z = np.median(depth_values)  # Use median depth
        
        # Convert image coordinates to camera coordinates
        x = (cx - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
        y = (cy - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
        
        # Create pose message
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "camera_color_optical_frame"
        
        # Set position (position in camera coordinate system)
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z - 0.05  # Slightly closer to object
        
        # Set orientation (grasp direction downward)
        q = tf.transformations.quaternion_from_euler(-np.pi/2, 0, 0)
        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]
        
        # Calculate a simple score
        score = 0.7  # Example score
        
        return pose_stamped, score
    
    def publish_default_grasp(self, header=None):
        """Publish default grasp pose"""
        default_pose = create_default_grasp()
        
        # If header provided, use it
        if (header):
            default_pose.header = header
        
        self.grasp_pub.publish(default_pose)
        rospy.loginfo("Published default grasp pose")
    
    def rotation_matrix_to_quaternion(self, rot_matrix):
        """Convert rotation matrix to quaternion"""
        # Removed tensorflow import, use numpy instead
        
        # Ensure input is a valid rotation matrix
        if isinstance(rot_matrix, np.ndarray):
            if (rot_matrix.shape != (3, 3)):
                rospy.logwarn(f"Rotation matrix has incorrect shape: {rot_matrix.shape}")
                # If not a 3x3 matrix, return default orientation (downward)
                return [0, 0, -0.707, 0.707]
        
        try:
            # Calculate quaternion elements
            trace = np.trace(rot_matrix)
            
            if (trace > 0):
                S = np.sqrt(trace + 1.0) * 2
                qw = 0.25 * S
                qx = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
                qy = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
                qz = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
            elif (rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]):
                S = np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
                qw = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
                qx = 0.25 * S
                qy = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
                qz = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
            elif (rot_matrix[1, 1] > rot_matrix[2, 2]):
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
            rospy.logerr(f"Quaternion conversion error: {e}")
            # Return default orientation (downward)
            return [0, 0, -0.707, 0.707]
    
    # Publish point cloud while publishing grasp markers
    def publish_grasp_cloud(self, grasp_results):
        """Publish grasp point cloud for visualization"""
        from sensor_msgs.msg import PointCloud2, PointField
        import sensor_msgs.point_cloud2 as pc2
        
        if not grasp_results:
            return
            
        # Create point cloud message
        cloud_msg = PointCloud2()
        cloud_msg.header.frame_id = "camera_color_optical_frame"
        cloud_msg.header.stamp = rospy.Time.now()
        
        # Extract all grasp points
        points = []
        for grasp in grasp_results:
            # Convert to metric units
            x, y, z = grasp['point'] / 1000.0
            # Set color based on score - high scores green, low scores red
            r, g, b = int(255 * (1.0 - grasp['score'])), int(255 * grasp['score']), 0
            points.append([x, y, z, r, g, b])
        
        # Create point cloud fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.UINT8, count=1),
            PointField(name='g', offset=13, datatype=PointField.UINT8, count=1),
            PointField(name='b', offset=14, datatype=PointField.UINT8, count=1),
        ]
        
        # Create point cloud
        cloud_msg = pc2.create_cloud(cloud_msg.header, fields, points)
        self.grasp_cloud_pub.publish(cloud_msg)

if __name__ == "__main__":
    try:
        graspnet_node = GraspNetROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass