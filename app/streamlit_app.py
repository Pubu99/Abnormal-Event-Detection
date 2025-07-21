import streamlit as st
from ultralytics import YOLO
import torch
import timm
import torchvision.transforms as transforms
import cv2
import tempfile
import numpy as np
import os
import time

from src.config import *

st.set_page_config(page_title="Abnormal Event Detection", layout="wide")
st.title("🚨 Abnormal Event Detection (YOLO + ConvNeXt)")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
use_webcam = st.checkbox("Use Webcam")

if uploaded_file or use_webcam:
    device = torch.device(DEVICE)

    yolo_model = YOLO(YOLO_MODEL)
    yolo_model.to(device)

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    fps_text = st.empty()
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        yolo_device_index = device.index if device.type == 'cuda' else -1
        results = yolo_model(frame, device=yolo_device_index, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()
            anomaly_label = CLASSES[pred_idx]
            is_anomaly = anomaly_label != "NormalVideos"

        color = (0, 0, 255) if is_anomaly else (0, 255, 0)

        for (box, cls_id, conf) in zip(boxes, cls_ids, confs):
            x1, y1, x2, y2 = box
            cls_name = yolo_model.names[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        display_text = "Normal" if not is_anomaly else f"Anomaly: {anomaly_label}"
        cv2.putText(frame, display_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        stframe.image(frame, channels="BGR")

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        fps_text.text(f"FPS: {fps:.2f}")

    cap.release()

else:
    st.info("Upload a video file or select 'Use Webcam' to start detection.")
