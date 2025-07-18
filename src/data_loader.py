import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class UCFDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for idx, cls in enumerate(classes):
            cls_folder = os.path.join(root_dir, cls)
            if not os.path.exists(cls_folder):
                continue
            for img_name in os.listdir(cls_folder):
                if img_name.endswith('.png'):
                    self.image_paths.append(os.path.join(cls_folder, img_name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

