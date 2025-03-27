# YOLOv8 GraspNet ROS

## Installation Guide

Create and use a virtual environment:

```bash
# Create virtual environment
python3 -m venv ~/python_envs/yolo_env

# Activate environment
source ~/python_envs/yolo_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install GraspNetAPI
cd ~/catkin_ws/src/
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install -e .

# Install GraspNet-Baseline
cd ~/catkin_ws/src/
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
pip install -e .
```

## Configuration

After installation, you need to modify certain paths and parameters to match your environment:

### Required Modifications

1. **Camera Topics**: Modify the following topics in launch files to match your camera setup:
   ```xml
   <param name="rgb_topic" value="/rgb/image_raw" />
   <param name="depth_topic" value="/depth_to_rgb/image_raw" />
   <param name="camera_info" value="/rgb/camera_info" />
   ```

2. **Model Paths**: Update the model paths in launch files and scripts:
   ```xml
   <!-- In launch files -->
   <param name="model_dir" value="$(find yolo_graspnet_ros)/models/graspnet" />
   <param name="model_path" value="$(find yolo_graspnet_ros)/models/yolov8m.pt" />
   ```
   
   ```python
   # In Python scripts (dl_model_utils.py)
   # Change this path to your GraspNet-Baseline installation directory
   graspnet_baseline_path = "/home/msi/yolo_3d_ws/src/graspnet-baseline"
   
   # Change this path to your GraspNetAPI installation directory
   graspnet_path = "/home/msi/yolo_3d_ws/src/graspnetAPI"
   ```

3. **Camera Frame ID**: Change the frame ID to match your camera's coordinate frame:
   ```python
   # In grasp_utils.py and other files
   frame_id = "camera_color_optical_frame"  # Modify based on your camera setup
   ```

4. **Camera Parameters**: If the automatic calibration fails, you may need to manually set camera intrinsics:
   ```python
   # In graspnet_generator.py
   self.camera_matrix = np.array([[550.0, 0, 320.0], [0, 550.0, 240.0], [0, 0, 1]])
   ```

5. **Point Cloud Processing Parameters**: Adjust processing parameters based on your point cloud density and object size:
   ```python
   # In pointcloud_utils.py
   # Adjust voxel_size parameter based on your point cloud density
   pcd = pcd.voxel_down_sample(voxel_size=0.01)
   
   # Adjust search radius and max_nn parameters for normal estimation
   pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
   
   # Adjust radius parameter based on your object size
   cv2.circle(mask, (int(x_center), int(y_center)), radius, 255, -1)
   ```

6. **Python Environment Path**: Update the Python path in launch files if your environment is different:
   ```xml
   launch-prefix="env PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$(env PYTHONPATH) $(env HOME)/python_envs/yolo_env/bin/python3"
   ```

## Commands

### Terminal 1
```sh
roslaunch azure_kinect_ros_driver driver.launch
```

Run your camera ROS driver

### Terminal 2
```sh
roslaunch yolo_graspnet_ros yolo_graspnet_generator_with_dl.launch
```

### Terminal 3
```sh
rostopic echo /best_grasp_pose
```
Check the best grasp pose.

## Launch Files

- `detector_3d.launch`: YOLOv8 3D object detection
- `yolo_graspnet_generator.launch`: Simplified version with YOLO and GraspNet
- `yolo_graspnet_generator_with_dl.launch`: Complete version with deep learning integration

## Models

- `.pt` files are YOLO model files.
- `.tar` files are GraspNet model files. The pre-trained weights from GitHub should work.

## Scripts

- `detector_3d_node.py`: YOLOv8 3D object detection, returns the 3D coordinates of objects relative to the camera.
- `graspnet_generator.py`: Simplified version for grasp pose generation.
- `graspnet_generator_with_dl.py`: Advanced version with GraspNet deep learning integration.

## System Requirements

### System Environment
- Ubuntu 20.04 (recommended)
- ROS Noetic
- Python 3.8+
- CUDA 11.x (recommended for GPU acceleration)

### Hardware Requirements
- NVIDIA GPU with 6GB+ VRAM (for deep learning models)
- 16GB+ RAM

### Pre-trained Models
- YOLOv8m.pt (object detection model)
- GraspNet pre-trained model (grasp pose generation)

## Special Notes

1. For virtual environment usage, ensure ROS-related packages are accessible:
   ```bash
   # Create a link to ROS Python packages in the virtual environment
   echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH" >> ~/python_envs/yolo_env/bin/activate
   ```

2. Adjust dependency versions according to your CUDA and PyTorch versions.

3. Ensure the camera is properly connected and configured, and the corresponding topics are published correctly before running.

4. For GraspNet:
   - This project includes a simplified implementation when GraspNet is not available
   - You can force the simplified implementation by setting:
     ```xml
     <param name="force_simplified" value="true" />
     ```
   - For full GraspNet functionality, you need to install GraspNet-Baseline and GraspNetAPI separately

5. Visualization:
   - RViz configuration is included for visualization
   - The system publishes grasp markers for the top candidates
   - You can adjust the number of displayed grasps in the code:
     ```python
     # In graspnet_generator.py
     for j, grasp in enumerate(valid_grasps[:10]):  # Only display top 10
     ```