#!/usr/bin/env python3
# filepath: d:\GitHub repo\yolov8_graspnet_ros\yolo_graspnet_ros\scripts\utils\setup_graspnet.py

import os
import sys
import importlib

# Try to find and import GraspNet modules
def test_graspnet_imports():
    # Add path
    # MODIFY: Change this path to your GraspNet-Baseline installation directory
    graspnet_path = "/home/msi/yolo_3d_ws/src/graspnet-baseline"
    sys.path.insert(0, graspnet_path)
    
    # Add submodule paths
    for subdir in ['models', 'utils', 'dataset']:
        subpath = os.path.join(graspnet_path, subdir)
        if os.path.exists(subpath):
            sys.path.insert(0, subpath)
            print(f"Added subpath: {subpath}")
    
    # Print current paths
    print("Current Python paths:")
    for p in sys.path:
        print(f"  {p}")
    
    # Try to import key modules
    modules_to_test = [
        'backbone', 
        'models.graspnet', 
        'models.loss', 
        'dataset.graspnet_dataset'
    ]
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"Successfully imported {module_name}")
        except ImportError as e:
            print(f"Cannot import {module_name}: {e}")
            
            # Try to find module files
            parts = module_name.split('.')
            if len(parts) > 1:
                base_name = parts[-1]
                for p in sys.path:
                    potential_file = os.path.join(p, f"{base_name}.py")
                    if os.path.exists(potential_file):
                        print(f"Found potential module file: {potential_file}")

if __name__ == "__main__":
    test_graspnet_imports()