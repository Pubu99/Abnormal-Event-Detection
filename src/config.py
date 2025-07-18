import os

DATA_DIR = "data/raw"
MODEL_DIR = "models/classifiers"
YOLO_MODEL = "yolov8n.pt"
DEVICE = "cuda" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu"

# Binary classification
CLASSES = ['NormalVideos', 'Anomaly']
NUM_CLASSES = len(CLASSES)
BINARY = True

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10
IMAGE_SIZE = (224, 224)

# For timm models
MODEL_NAME = "convnext_tiny.fb_in22k"
