import cv2 as cv
from ultralytics import YOLO

video = r'C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_24.mp4'
cap = cv.VideoCapture(video)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv.imshow('Frame', frame)
    if cv.waitKey(1) == 27:
        break
cap.release()
cv.destroyAllWindows()

cv.imshow('Image', frame)
cv.waitKey(0)

# Perform tracking with the model
model = YOLO('best.pt')
results = model.track(frame, conf=0.4, iou=0.6)
print(results)
print(len(results))
print(results[0].boxes)
# results = model.track(video, conf=0.4, iou=0.6, show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker
