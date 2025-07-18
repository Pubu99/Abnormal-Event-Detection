import os
import torch

DATA_DIR = "data/raw"
MODEL_DIR = "models/classifiers"
YOLO_MODEL = "yolov8n.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Multi-class classification (14 anomaly types + Normal)
CLASSES = ['NormalVideos', 'Explosion', 'Fighting', 'Robbery', 'Shooting',
           'Stealing', 'Vandalism', 'Arson', 'Assault', 'Burglary',
           'Accident', 'Trespassing', 'Riot', 'WeaponThreat', 'SuspiciousActivity']
NUM_CLASSES = len(CLASSES)

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10
IMAGE_SIZE = (224, 224)
MODEL_NAME = "convnext_tiny.fb_in22k"
