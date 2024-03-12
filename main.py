from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/my_yolov8.yaml")
model.train(**{'cfg':'ultralytics/cfg/default.yaml'})