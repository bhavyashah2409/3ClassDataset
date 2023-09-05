from ultralytics import YOLO

if __name__ == '__main__':
    model_type = 'runs\detect\yolov8l_3c_500e\weights\last.pt'
    model = YOLO(model_type)
    # model.train(data='custom_data.yaml', batch=4, workers=2, imgsz=640, epochs=500, name='yolov8l_3c_500e', device=0, patience=50, verbose=True)
    model.train(resume=True)
