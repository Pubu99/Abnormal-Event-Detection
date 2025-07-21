import os
import torch

# Directory paths
DATA_DIR = "data/raw"
MODEL_DIR = "models/classifiers"
YOLO_MODEL = "yolov8n.pt"

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class labels (make sure these exactly match your dataset folders)
CLASSES = [
    "NormalVideos", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism"
]
NUM_CLASSES = len(CLASSES)

# Training configuration
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10
IMAGE_SIZE = (224, 224)

# Model backbone
MODEL_NAME = "convnext_tiny.fb_in22k"
