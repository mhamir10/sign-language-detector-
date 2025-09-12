from ultralytics import YOLO
import os
import numpy as np
model_path = 'runs/detect/bangla_sign_detector_v1_final11/weights/best.pt'
data_yaml_path = 'D:/university/8th semmester/machine learning/bangla sign language detector/data.yaml'

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please ensure the path to the trained model is correct.")
    exit()

if not os.path.exists(data_yaml_path):
    print(f"Error: data.yaml file not found at {data_yaml_path}")
    print("Please ensure the path to the data.yaml file is correct.")
    exit()

model = YOLO(model_path)

print("Starting model evaluation on the test set...")
try:
    metrics = model.val(data=data_yaml_path)

    print(f"\nMetrics object type: {type(metrics)}")
    print(f"Metrics object content (dir): {dir(metrics)}")

    if hasattr(metrics, 'box') and metrics.box is not None:
        print("\nEvaluation Results:")

        def get_scalar_metric(metric_attr):
            if metric_attr is None:
                return float('nan')

            if isinstance(metric_attr, np.ndarray):
                if metric_attr.size > 0:
                    return float(metric_attr.item(0))
                else:
                    return float('nan')

            try:
                return float(metric_attr)
            except (ValueError, TypeError):
                return float('nan')


        precision = get_scalar_metric(metrics.box.p)
        recall = get_scalar_metric(metrics.box.r)
        map50 = get_scalar_metric(metrics.box.map50)
        map_full = get_scalar_metric(metrics.box.map)

        print(f"Precision (P): {precision:.4f}")
        print(f"Recall (R): {recall:.4f}")
        print(f"mAP50: {map50:.4f}")
        print(f"mAP50-95: {map_full:.4f}")
    else:
        print("\nWarning: 'metrics.box' attribute not found or is None.")
        print("This may indicate that the model did not detect any objects in the test set.")
        print("If your test set contains images, please verify model training or dataset labeling.")

except Exception as e:
    print(f"\nAn error occurred while running the evaluation: {e}")
    print("Please ensure your test dataset is loaded correctly and contains label files.")

print("Model evaluation complete!")
