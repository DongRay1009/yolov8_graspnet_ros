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

from utils.bridge_utils import get_bridge
from utils.pointcloud_utils import prepare_pointcloud_for_graspnet
from utils.dl_model_utils import load_graspnet_model, create_camera_info, predict_grasps
from utils.grasp_utils import create_default_grasp
from utils.visualization import create_grasp_marker

class GraspNetROS:
    def __init__(self):
        """Initialize GraspNet ROS node"""
        rospy.init_node('graspnet_ros')

        # MODIFY: Change these topic names according to your camera setup
        self.rgb_topic = rospy.get_param('~rgb_topic', '/rgb/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/depth_to_rgb/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info', '/rgb/camera_info')
        self.detection_topic = rospy.get_param('~detection_topic', '/object_poses')
        
        # MODIFY: Set GraspNet model path according to your setup
        self.model_dir = rospy.get_param('~model_dir', 
                                        '/home/msi/yolo_3d_ws/src/yolo_graspnet_ros/models/graspnet')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint.tar')
        
        rospy.loginfo(f"Using model path: {self.checkpoint_path}")
        
        self.bridge = get_bridge()
        
        self.grasp_pub = rospy.Publisher('/best_grasp_pose', PoseStamped, queue_size=1)
        self.all_grasps_pub = rospy.Publisher('/all_grasp_poses', PoseArray, queue_size=1)
        self.debug_pub = rospy.Publisher('/grasp_debug', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/grasp_markers', MarkerArray, queue_size=1)
        self.grasp_cloud_pub = rospy.Publisher('/grasp_cloud', PointCloud2, queue_size=1)
        
        self.rgb_msg = None
        self.depth_msg = None
        self.poses_msg = None
        
        self.rgb_sub = rospy.Subscriber(
            self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self.depth_callback, queue_size=1)
        self.detection_sub = rospy.Subscriber(
            self.detection_topic, PoseArray, self.detection_callback, queue_size=1)
        
        self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_callback)
        
        self.camera_info = None
        self.camera_matrix = None
        self.get_camera_info()
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Load GraspNet model
        self.net, self.pred_decode, self.device = load_graspnet_model(self.checkpoint_path)
        self.model_loaded = self.net is not None
        
        rospy.loginfo("GraspNet ROS node initialized")
    
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
            rospy.logerr("Unable to get camera information")
            # MODIFY: Default camera matrix if your camera is different
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
        """Object detection results callback function"""
        self.poses_msg = msg
        rospy.loginfo_throttle(2.0, f"Received detection result with {len(msg.poses)} objects")
    
    def timer_callback(self, event):
        """Timer callback function to process collected messages"""
        # First ensure model is loaded
        if not hasattr(self, 'model_loaded') or self.model_loaded is None:
            rospy.loginfo_throttle(2.0, "Model not yet loaded, skipping processing")
            return
        
        # Check if all necessary messages have been received
        if self.rgb_msg is None or self.depth_msg is None:
            return
            
        # Can also try generating grasps without detection results
        if self.poses_msg is None:
            rospy.loginfo_throttle(2.0, "No object detection results received, using full scene")
            
        # Copy messages to avoid modification by other callbacks during processing
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
                rospy.loginfo("No detection results, will generate grasp poses for full scene")
                pose_count = 0
            else:
                pose_count = len(poses_msg.poses)
                rospy.loginfo(f"Number of detected targets: {pose_count}")
            
            # Convert images
            try:
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
                depth = self.bridge.imgmsg_to_cv2(depth_msg)
                rospy.loginfo(f"Image conversion successful: RGB {rgb.shape}, depth {depth.shape}")
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
            
            # Process differently based on whether there are detection results
            if poses_msg is None or pose_count == 0:
                # No detection results, generate grasp poses for the full scene
                if self.model_loaded:
                    # Use the already loaded model, don't reload
                    grasp_results = predict_grasps(self.net, self.pred_decode, self.device, rgb, depth, camera_info)
                    if grasp_results and len(grasp_results) > 0:
                        rospy.loginfo(f"Generated {len(grasp_results)} grasp candidates for full scene")
                        self.process_grasp_results(grasp_results)
                    else:
                        rospy.logwarn("No valid grasp candidates generated")
                        self.publish_default_grasp()
                else:
                    rospy.logwarn("Model not loaded, cannot generate grasp poses")
                    self.publish_default_grasp()
            else:
                # With detection results, generate grasp poses for each object
                for i, pose in enumerate(poses_msg.poses):
                    position = pose.position
                    rospy.loginfo(f"Generating grasp pose for object {i+1}, position: ({position.x:.3f}, {position.y:.3f}, {position.z:.3f})")
                    
                    # Generate grasp pose
                    if self.model_loaded:
                        grasp_results = predict_grasps(self.net, self.pred_decode, self.device, rgb, depth, camera_info)
                        
                        if grasp_results and len(grasp_results) > 0:
                            rospy.loginfo(f"Generated {len(grasp_results)} grasp candidates for object {i+1}")
                            self.process_grasp_results(grasp_results)
                        else:
                            rospy.logwarn(f"No valid grasp candidates generated for object {i+1}")
                            
                    # Only process the first object
                    break
        
        except Exception as e:
            rospy.logerr(f"Message processing exception: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            
    def process_grasp_results(self, grasp_results):
        """Process grasp results and publish"""
        try:
            # Filter valid grasp poses
            valid_grasps = self.filter_valid_grasps(grasp_results)
            
            # Publish best grasp pose
            best_grasp = valid_grasps[0]  # Results already sorted by score
            best_pose = PoseStamped()
            # MODIFY: Change this frame_id if your camera frame is different
            best_pose.header.frame_id = "camera_color_optical_frame"
            best_pose.header.stamp = rospy.Time.now()
            
            # Set position (convert to meters)
            best_pose.pose.position.x = float(best_grasp['point'][0]) / 1000.0
            best_pose.pose.position.y = float(best_grasp['point'][1]) / 1000.0
            best_pose.pose.position.z = float(best_grasp['point'][2]) / 1000.0
            
            # Set orientation
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
                # Use default orientation if conversion fails
                best_pose.pose.orientation.w = 1.0
            
            # Publish best grasp pose
            self.grasp_pub.publish(best_pose)
            rospy.loginfo(f"Publishing best grasp pose: " +
                          f"position=({best_pose.pose.position.x:.3f}, {best_pose.pose.position.y:.3f}, {best_pose.pose.position.z:.3f}), " +
                          f"score={best_grasp['score']:.3f}")
            
            # Create and publish marker array
            markers = MarkerArray()
            for j, grasp in enumerate(valid_grasps[:10]):  # Only show top 10
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
                rospy.loginfo(f"Published {len(markers.markers)} grasp markers")
                
            # Publish grasp point cloud
            self.publish_grasp_cloud(grasp_results)
                
        except Exception as e:
            rospy.logerr(f"Failed to process grasp results: {e}")
            self.publish_default_grasp()
    
    def filter_valid_grasps(self, grasp_results):
        """Filter valid grasp poses"""
        valid_grasps = []
        for grasp in grasp_results:
            point = grasp['point']
            # Example: filter out points that are too far or too close
            distance = np.linalg.norm(point)
            if 0.05 < distance < 1.5:  # Unit: meters
                valid_grasps.append(grasp)
        
        return valid_grasps if valid_grasps else grasp_results  # Return original results if all filtered out
    
    def create_pose_stamped(self, pose):
        """Create PoseStamped message from Pose"""
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        # MODIFY: Change this frame_id if your camera frame is different
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
        # MODIFY: Change this frame_id if your camera frame is different
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
    
    def publish_default_grasp(self):
        """Publish default grasp pose"""
        default_pose = create_default_grasp()
        self.grasp_pub.publish(default_pose)
        rospy.loginfo("Publishing default grasp pose")
    
    def rotation_matrix_to_quaternion(self, rot_matrix):
        """Convert rotation matrix to quaternion"""
        import tensorflow as tf
        
        # Ensure input is a valid rotation matrix
        if isinstance(rot_matrix, np.ndarray):
            if rot_matrix.shape != (3, 3):
                rospy.logwarn(f"Rotation matrix has incorrect shape: {rot_matrix.shape}")
                # Return default orientation (downward) if not a 3x3 matrix
                return [0, 0, -0.707, 0.707]
        
        try:
            # Calculate quaternion elements
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
            rospy.logerr(f"Quaternion conversion error: {e}")
            # Return default orientation (downward)
            return [0, 0, -0.707, 0.707]
    
    def publish_grasp_cloud(self, grasp_results):
        """Publish grasp point cloud for visualization"""
        from sensor_msgs.msg import PointCloud2, PointField
        import sensor_msgs.point_cloud2 as pc2
        
        if not grasp_results:
            return
            
        # Create point cloud message
        cloud_msg = PointCloud2()
        # MODIFY: Change this frame_id if your camera frame is different
        cloud_msg.header.frame_id = "camera_color_optical_frame"
        cloud_msg.header.stamp = rospy.Time.now()
        
        # Extract all grasp points
        points = []
        for grasp in grasp_results:
            # Convert to meter units
            x, y, z = grasp['point'] / 1000.0
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