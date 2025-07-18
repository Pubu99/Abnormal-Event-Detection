from ultralytics import YOLO
import torch
import timm
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import time

from src.config import *

def hybrid_infer(source=0):
    device = torch.device(DEVICE)

    # Load YOLO and move to GPU
    yolo_model = YOLO(YOLO_MODEL)
    yolo_model.to(device)

    # Load ConvNeXt classifier (multi-class)
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{MODEL_NAME}_epoch_{EPOCHS}.pth"), map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    cap = cv2.VideoCapture(source)

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # YOLO object detection
        results = yolo_model(frame, device=0, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        # Frame → Classifier (multi-class)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            anomaly_label = CLASSES[pred_idx]

        # Choose color
        is_anomaly = anomaly_label != 'NormalVideos'
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)

        # Draw YOLO boxes
        for (box, cls_id, conf) in zip(boxes, cls_ids, confs):
            x1, y1, x2, y2 = box
            cls_name = yolo_model.names[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw anomaly classification result
        display_text = "Normal" if not is_anomaly else f"Anomaly: {anomaly_label}"
        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # FPS Calculation
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("YOLO + ConvNeXt Anomaly Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    hybrid_infer()
