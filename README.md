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
- `yolo_graspnet_generator_with_dl.launch`: YOLO and graspnet

## Models

- `.pt` files are YOLO model files.
- `.tar` files are GraspNet model files. The pre-trained weights from GitHub should work.

## Scripts

- `detector_3d_node.py`: YOLOv8 3D object detection, returns the 3D coordinates of objects relative to the camera.
- `graspnet_generator_with_dl.py`: Version with GraspNet.

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