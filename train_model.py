from ultralytics import YOLO
model = YOLO('yolov8n.pt')

print("model training started...")
results = model.train(data='D:/university/8th semmester/machine learning/bangla sign language detector/data.yaml',
                      epochs=100,
                      batch=16,
                      imgsz=640,
                      name='bangla_sign_detector_v1_final')
print("Model training completed!")