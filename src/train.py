# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os
import timm

from src.config import *
from src.data_loader import UCFBinaryDataset

def train():
    device = torch.device(DEVICE)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    dataset = UCFBinaryDataset("data/processed", transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=4)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels.int()).sum().item()
            total_loss += loss.item() * imgs.size(0)

        print(f"Train Loss: {total_loss/train_size:.4f} | Train Acc: {correct/train_size:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                preds = (torch.sigmoid(outputs) > 0.5).int()
                val_correct += (preds == labels.int()).sum().item()
                val_loss += loss.item() * imgs.size(0)

        print(f"Val Loss: {val_loss/val_size:.4f} | Val Acc: {val_correct/val_size:.4f}")

        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{MODEL_NAME}_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    train()
