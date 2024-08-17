from roboflow import Roboflow
from ultralytics import YOLO
import ultralytics
import os

from roboflow import Roboflow
rf = Roboflow(api_key="jIXqeO4L77RymWj9htLH")
project = rf.workspace("capstone-project-rmit").project("waste-detection-z1ra8")
version = project.version(5)
dataset = version.download("yolov8-obb")

def update_yaml_paths(yaml_file, new_base_path):
    # Read the existing YAML file line by line
    with open(yaml_file, 'r') as file:
        lines = file.readlines()

    # Create a temporary file to store the modified content
    tmp_yaml_file = 'tmp_data.yaml'
    with open(tmp_yaml_file, 'w') as tmp_file:
        for line in lines:
            # Update the path, train, val, and test lines
            if line.startswith('path:'):
                tmp_file.write(f'path: {new_base_path}\n')
            elif line.startswith('train:'):
                tmp_file.write(f'train: {os.path.join(new_base_path, "train/images")}\n')
            elif line.startswith('val:'):
                tmp_file.write(f'val: {os.path.join(new_base_path, "valid/images")}\n')
            elif line.startswith('test:'):
                tmp_file.write(f'test: {os.path.join(new_base_path, "test/images")}\n')
            else:
                tmp_file.write(line)

    # Replace the original file with the temporary file
    os.replace(tmp_yaml_file, yaml_file)
    print(f"Updated paths in {yaml_file} to base path: {new_base_path}")

    # Remove the temporary file (optional since it has already been replaced)
    if os.path.exists(tmp_yaml_file):
        os.remove(tmp_yaml_file)

# Example usage
yaml_file =  str(dataset.location) + '/data.yaml'
new_base_path =  str(dataset.location)
update_yaml_paths(yaml_file, new_base_path)


# model = YOLO("yolov8n-obb") # 3.1M params
# model = YOLO("yolov8s-obb") # 11.4M params
# model = YOLO("yolov8m-obb") # 26.4M params
# model = YOLO("yolov8l-obb") # 44.5M params
model = YOLO("yolov8x-obb") # 69.5M params

save_dir_ = "./"

train_result = model.train(data=os.path.join(dataset.location, "data.yaml"),
                           epochs=160,
                           imgsz=640,
                           plots=True,
                           save_period=50,
                           save_dir=save_dir_,
                           batch=-1,
                           device=[0,1,2,3])