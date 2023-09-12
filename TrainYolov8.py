from ultralytics import YOLO

if __name__ == '__main__':
    model_type = 'yolov8l.pt'
    model = YOLO(model_type)
    model.train(data='custom_data.yaml', batch=8, workers=2, imgsz=640, epochs=50, name='yolov8l_pretrained', device=0, patience=30, verbose=True, pretrained=True)
    # model.train(resume=True)
