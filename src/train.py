import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
import timm

from src.config import *
from src.data_loader import UCFAugmentedDataset
from src.augmentations import get_train_transforms, get_val_transforms

def get_sampler(dataset):
    # Optional: handle class imbalance
    label_count = [0] * NUM_CLASSES
    for _, label in dataset:
        label_count[label] += 1

    weights = [1.0 / label_count[label] for _, label in dataset]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

def train():
    print("CUDA Available:", torch.cuda.is_available())
    print("Device:", DEVICE)
    device = torch.device(DEVICE)

    # Use advanced augmentations
    train_ds = UCFAugmentedDataset(os.path.join(DATA_DIR, "Train"), transform=get_train_transforms(IMAGE_SIZE))
    val_ds = UCFAugmentedDataset(os.path.join(DATA_DIR, "Test"), transform=get_val_transforms(IMAGE_SIZE))

    train_sampler = get_sampler(train_ds)  # Optional: comment if no imbalance

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = torch.cuda.amp.GradScaler()  # For mixed precision

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0

        loop = tqdm(train_loader, desc=f"🔁 Epoch {epoch + 1}/{EPOCHS}")

        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)
        train_acc = correct / len(train_ds)
        print(f"✅ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)
        print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step()

        # Save best model only
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 New best model saved: {best_model_path}")

if __name__ == "__main__":
    train()
