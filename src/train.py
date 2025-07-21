import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import timm

from src.config import *
from src.data_loader import UCFAugmentedDataset

def train():
    print("CUDA Available:", torch.cuda.is_available())
    if DEVICE == "cuda":
        print("🚀 Training on GPU:", torch.cuda.get_device_name(0))
    else:
        print("🧠 Training on CPU")

    device = torch.device(DEVICE)

    # ✅ Improved data augmentation
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_ds = UCFAugmentedDataset(os.path.join(DATA_DIR, "Train"), transform)
    val_ds = UCFAugmentedDataset(os.path.join(DATA_DIR, "Test"), transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)

    # ✅ Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(MODEL_DIR, exist_ok=True)

    best_val_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0

        print(f"\n🔁 Epoch {epoch + 1}/{EPOCHS}")
        for imgs, labels in tqdm(train_loader, desc="🧪 Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * imgs.size(0)

        train_loss = total_loss / len(train_ds)
        train_acc = correct / len(train_ds)
        print(f"✅ Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")

        # Evaluation
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)
        print(f"📊 Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        # Save model if improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"💾 Best model saved: {model_path}")

        # Optional: Save each epoch
        model_path_epoch = os.path.join(MODEL_DIR, f"{MODEL_NAME}_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path_epoch)

        # Optional: Early stopping (commented)
        # if val_acc < best_val_acc - 0.01:
        #     print("⚠️ Early stopping - validation accuracy dropped.")
        #     break

if __name__ == "__main__":
    train()
