# src/evaluate.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
import timm
from tqdm import tqdm

from src.config import *
from src.data_loader import UCFBinaryDataset

def evaluate():
    device = torch.device(DEVICE)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    dataset = UCFBinaryDataset("data/processed", transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{MODEL_NAME}_epoch_{EPOCHS}.pth"), map_location=device))
    model = model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy().flatten()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    print(classification_report(y_true, y_pred, target_names=CLASSES))

if __name__ == "__main__":
    evaluate()
