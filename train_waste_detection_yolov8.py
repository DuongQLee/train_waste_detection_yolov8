from roboflow import Roboflow
from ultralytics import YOLO
import ultralytics
import os
rf = Roboflow(api_key="jIXqeO4L77RymWj9htLH")
project = rf.workspace("capstone-project-rmit").project("waste-detection-z1ra8")
version = project.version(3)
dataset = version.download("yolov8-obb")

# model = YOLO("yolov8n-obb") # 3.1M params
# model = YOLO("yolov8s-obb") # 11.4M params
# model = YOLO("yolov8m-obb") # 26.4M params
# model = YOLO("yolov8l-obb") # 44.5M params
model = YOLO("yolov8x-obb") # 69.5M params

save_dir_ = "./"

train_result = model.train(data=os.path.join(dataset.location, "data.yaml"),
                           epochs=300,
                           imgsz=640,
                           plots=True,
                           save_period=50,
                           save_dir=save_dir_,
                           device=[0])