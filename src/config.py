import os

DATA_DIR = "data/raw"
MODEL_DIR = "models/classifiers"
YOLO_MODEL = "yolov8n.pt"  # can replace with yolov8s.pt for higher accuracy
DEVICE = "cuda" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu"

NUM_CLASSES = 14
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10
IMAGE_SIZE = (224, 224)

CLASSES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
    'Fighting', 'NormalVideos', 'RoadAccidents', 'Robbery',
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]

