#!/home/msi/python_envs/yolo_env/bin/python3
# filepath: /home/msi/yolo_3d_ws/src/yolo_graspnet_ros/scripts/utils/setup_graspnet.py

import os
import sys
import importlib

# 尝试找到并导入GraspNet模块
def test_graspnet_imports():
    # 添加路径
    graspnet_path = "/home/msi/yolo_3d_ws/src/graspnet-baseline"
    sys.path.insert(0, graspnet_path)
    
    # 添加子模块路径
    for subdir in ['models', 'utils', 'dataset']:
        subpath = os.path.join(graspnet_path, subdir)
        if os.path.exists(subpath):
            sys.path.insert(0, subpath)
            print(f"已添加子路径: {subpath}")
    
    # 打印当前路径
    print("当前Python路径:")
    for p in sys.path:
        print(f"  {p}")
    
    # 尝试导入关键模块
    modules_to_test = [
        'backbone', 
        'models.graspnet', 
        'models.loss', 
        'dataset.graspnet_dataset'
    ]
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"成功导入 {module_name}")
        except ImportError as e:
            print(f"无法导入 {module_name}: {e}")
            
            # 尝试查找模块文件
            parts = module_name.split('.')
            if len(parts) > 1:
                base_name = parts[-1]
                for p in sys.path:
                    potential_file = os.path.join(p, f"{base_name}.py")
                    if os.path.exists(potential_file):
                        print(f"找到可能的模块文件: {potential_file}")

if __name__ == "__main__":
    test_graspnet_imports()