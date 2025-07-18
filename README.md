# Abnormal Event Detection in Multi-Camera Surveillance Systems

## Overview

This project implements a **real-time abnormal event detection system** using **YOLOv8** for object detection and **ResNet-50** for anomaly classification. It is designed for surveillance camera feeds and supports video files, webcam, and multi-camera extension.

Key features:

- Detects suspicious/abnormal events across 14 categories (e.g., Arson, Robbery, Shooting).
- Visualizes detections with green bounding boxes for normal events, red boxes for anomalies.
- Real-time inference using a hybrid YOLO + ResNet pipeline.
- Streamlit frontend for easy demo and testing.

---

## Project Structure

```
abnormal-event-detection/
├── data/                    # Raw and processed datasets
├── models/                  # YOLO and ResNet models/checkpoints
├── src/                     # Source code (training, inference, utils)
├── app/                     # Streamlit frontend application
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── train_config.yaml       # (Optional) Training configs
```

---

## Setup Instructions

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Prepare dataset**

Download and place UCF Crime dataset images in `data/raw/<class_name>/`.

3. **Preprocess dataset**

Resize images to 224x224 for ResNet:

```bash
python -m src.preprocess
```

4. **Train ResNet classifier**

```bash
python -m src.train
```

5. **Evaluate trained model (optional)**

```bash
python -m src.evaluate
```

6. **Run YOLO-only detection (optional)**

```bash
python -m src.yolo_infer
```

7. **Run hybrid YOLO + ResNet inference**

```bash
python -m src.infer
```

8. **Run frontend**

```bash
streamlit run app/streamlit_app.py
```

---

## Usage

- Upload video files or select "Use Webcam" in the frontend to start detection.
- View bounding boxes and anomaly classification live.
- Press `q` to quit any OpenCV window.

---

## Notes

- You can change hyperparameters and paths in `src/config.py`.
- For better accuracy, try larger YOLO models (yolov8s.pt) and train for more epochs.
- Multi-camera support can be added by extending the video capture in `src/infer.py` and frontend.

---

## References

- UCF Crime Dataset: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
- YOLOv8: Ultralytics GitHub https://github.com/ultralytics/ultralytics
- ResNet paper: He et al., Deep Residual Learning for Image Recognition, 2015

---

# 🙌 Summary

- Organize folder → Install packages → Prepare data → Preprocess → Train → Evaluate → Infer → Frontend  
- Use your trained models for inference and demo  
- Easily tweak config for more epochs, batch size, or model variants
