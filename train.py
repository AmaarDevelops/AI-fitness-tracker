from ultralytics import YOLO

model  = YOLO('yolov8n-pose.pt')

results = model.train(
    data='coco8-pose.yaml',
    imgsz=640,
    epochs=5,
    batch=4,
    name='fitness_tracker_yolov8'
)
