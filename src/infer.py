from ultralytics import YOLO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import os
from src.config import *
import numpy as np

def hybrid_infer(source=0):
    device = torch.device(DEVICE)

    # Load YOLOv8
    yolo_model = YOLO(YOLO_MODEL)

    # Load ResNet
    resnet = models.resnet50(pretrained=False)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, NUM_CLASSES)
    resnet.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"resnet50_epoch_{EPOCHS}.pth"), map_location=device))
    resnet = resnet.to(device)
    resnet.eval()

    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Failed to open video source: {source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video ended or failed to grab frame.")
            break

        # ---------------- YOLOv8 Detection ----------------
        results = yolo_model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        # ---------------- ResNet Classification ----------------
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = resnet(img_tensor)  # shape: (1, NUM_CLASSES)
            _, pred = torch.max(outputs, 1)
            pred_class = CLASSES[pred.item()]
        
        # Set display color
        is_anomaly = pred_class != "NormalVideos"
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        anomaly_label = "Anomaly" if is_anomaly else "Normal"

        # Draw YOLO boxes
        for (box, cls_id, conf) in zip(boxes, cls_ids, confs):
            x1, y1, x2, y2 = box
            label = f"{yolo_model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show classification result
        cv2.putText(frame, f"Anomaly: {anomaly_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        cv2.imshow("YOLO + ResNet Abnormal Event Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    hybrid_infer()
