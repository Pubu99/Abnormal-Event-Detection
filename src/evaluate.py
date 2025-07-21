import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import timm
import numpy as np
from tqdm import tqdm

from src.config import *
from src.data_loader import UCFAugmentedDataset

def find_best_model(model_dir):
    """
    Automatically find the best epoch based on model filename like '..._epoch_#.pth'
    """
    weights = [f for f in os.listdir(model_dir) if f.endswith('.pth') and '_epoch_' in f]
    if not weights:
        raise FileNotFoundError("❌ No .pth model files found in model directory.")

    # Sort by epoch number
    weights = sorted(weights, key=lambda x: int(x.split('_epoch_')[-1].split('.')[0]))
    best_model_path = os.path.join(model_dir, weights[-1])  # Last epoch (or pick best manually)
    print(f"🔍 Using Model: {weights[-1]}")
    return best_model_path

def evaluate():
    device = torch.device(DEVICE)
    print("📦 Device:", device)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load Test Dataset
    dataset = UCFAugmentedDataset(os.path.join(DATA_DIR, "Test"), transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load Model
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model_path = find_best_model(MODEL_DIR)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluation
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="🧪 Evaluating"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Filter only used classes
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    used_classes = [CLASSES[i] for i in unique_labels]

    # 🔷 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    acc = accuracy_score(y_true, y_pred)

    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=used_classes, yticklabels=used_classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Accuracy: {acc:.4f})")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    print("✅ Confusion matrix saved: reports/confusion_matrix.png")
    plt.show()

    # 🔷 Classification Report
    report = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=used_classes,
        zero_division=0
    )
    print("🔎 Classification Report:\n")
    print(report)

    with open("reports/classification_report.txt", "w") as f:
        f.write(report)
    print("✅ Classification report saved: reports/classification_report.txt")

if __name__ == "__main__":
    evaluate()
