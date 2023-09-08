from ultralytics import YOLO

# Load an official or custom model
model = YOLO(r'runs\detect\yolov8l_3c_500e\weights\best.pt')

video = r'C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_8.mp4'

# Perform tracking with the model
results = model.track(video, conf=0.4, iou=0.6, show=True)  # Tracking with default tracker
results = model.track(video, conf=0.4, iou=0.6, show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker
