import streamlit as st
from ultralytics import YOLO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import tempfile
import numpy as np
import os

from src.config import *

st.set_page_config(page_title="Abnormal Event Detection", layout="wide")
st.title("🚨 Abnormal Event Detection (YOLO + ResNet)")

# Upload video or use webcam checkbox
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
use_webcam = st.checkbox("Use Webcam")

if uploaded_file or use_webcam:
    # Load YOLO model (lightweight by default)
    yolo_model = YOLO(YOLO_MODEL)

    # Load ResNet classifier
    device = torch.device(DEVICE)
    resnet = models.resnet50(pretrained=False)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, NUM_CLASSES)
    model_path = os.path.join(MODEL_DIR, f"resnet50_epoch_{EPOCHS}.pth")
    resnet.load_state_dict(torch.load(model_path, map_location=device))
    resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Setup video capture
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    fps_text = st.empty()

    frame_count = 0
    import time
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = yolo_model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        # ResNet classification on full frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = resnet(img_tensor)
            _, pred = torch.max(outputs, 1)
            anomaly_label = CLASSES[pred.item()]

        # Set box color: green normal, red anomaly
        color = (0, 255, 0) if anomaly_label == "NormalVideos" else (0, 0, 255)

        # Draw bounding boxes and labels
        for (box, cls_id, conf) in zip(boxes, cls_ids, confs):
            x1, y1, x2, y2 = box
            cls_name = yolo_model.names[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display anomaly label on frame
        cv2.putText(frame, f"Anomaly: {anomaly_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        # Show frame in streamlit
        stframe.image(frame, channels="BGR")

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        fps_text.text(f"FPS: {fps:.2f}")

    cap.release()
else:
    st.info("Upload a video file or select 'Use Webcam' to start detection.")
