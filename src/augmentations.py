from torchvision import transforms

def get_train_transforms(image_size=(224,224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.05,0.05)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.02,0.1)),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def get_val_transforms(image_size=(224,224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
