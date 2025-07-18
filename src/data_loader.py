# src/data_loader.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.config import CLASSES

class UCFBinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for cls in os.listdir(root_dir):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            label = 0 if cls == 'NormalVideos' else 1
            for img_file in os.listdir(cls_path):
                if img_file.endswith('.png'):
                    self.image_paths.append(os.path.join(cls_path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
