from ultralytics import YOLO
import cv2
import os
import yaml

model_path = 'runs/detect/bangla_sign_detector_v1_final11/weights/best.pt'

data_yaml_path = 'D:/university/8th semmester/machine learning/bangla sign language detector/data.yaml'

if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}")
    exit()

if not os.path.exists(data_yaml_path):
    print(f"ERROR: data.yaml file not found at {data_yaml_path}")
    exit()

try:
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config['names']
except Exception as e:
    print(f"ERROR: Could not load class names from data.yaml: {e}")
    exit()

model = YOLO(model_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("WEBCAM_STARTED")

# Create a window to display the webcam feed
cv2.namedWindow("Sign Detection Live", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame from webcam.")
        break

    results = model.predict(source=frame, conf=0.5, verbose=False, show=False)

    detected_signs = []
    annotated_frame = frame.copy()

    for r in results:

        annotated_frame = r.plot()

        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            detected_sign_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_Sign_{class_id}"

            detected_signs.append(f"{detected_sign_name} ({confidence:.2f})")

    if detected_signs:
        print(f"DETECTED_SIGNS:{';'.join(detected_signs)}")
    else:
        print("DETECTED_SIGNS:None")

    cv2.imshow("Sign Detection Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("WEBCAM_STOPPED")
