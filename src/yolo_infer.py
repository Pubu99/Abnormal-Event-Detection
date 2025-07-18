from ultralytics import YOLO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from src.config import *
import numpy as np

def hybrid_infer(source=0):  # source = 0 for webcam, or video path
    device = torch.device(DEVICE)

    # Load YOLO
    yolo_model = YOLO(YOLO_MODEL)

    # Load ResNet
    resnet = models.resnet50(pretrained=False)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, NUM_CLASSES)
    resnet.load_state_dict(torch.load(f"{MODEL_DIR}/resnet50_epoch_{EPOCHS}.pth", map_location=device))
    resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = yolo_model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        # ResNet classification
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = resnet(img_tensor)
            _, pred = torch.max(outputs, 1)
            anomaly_label = CLASSES[pred.item()]

        # Determine box color
        if anomaly_label == "NormalVideos":
            color = (0, 255, 0)  # green
        else:
            color = (0, 0, 255)  # red

        # Draw boxes
        for (box, cls_id, conf) in zip(boxes, cls_ids, confs):
            x1, y1, x2, y2 = box
            cls_name = yolo_model.names[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Show anomaly label on frame
        cv2.putText(frame, f"Anomaly: {anomaly_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("YOLO + ResNet Abnormal Event Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    hybrid_infer()

