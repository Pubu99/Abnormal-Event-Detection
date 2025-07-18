from PIL import Image
import os
from tqdm import tqdm

def resize_images(input_dir, output_dir, size=(224,224)):
    os.makedirs(output_dir, exist_ok=True)
    for cls in os.listdir(input_dir):
        cls_input_path = os.path.join(input_dir, cls)
        cls_output_path = os.path.join(output_dir, cls)
        os.makedirs(cls_output_path, exist_ok=True)
        if not os.path.isdir(cls_input_path):
            continue
        img_files = [f for f in os.listdir(cls_input_path) if f.endswith('.png')]
        for img_file in tqdm(img_files, desc=f"Processing {cls}"):
            img_path = os.path.join(cls_input_path, img_file)
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize(size)
            img_resized.save(os.path.join(cls_output_path, img_file))

if __name__ == "__main__":
    resize_images("data/raw", "data/processed", size=(224,224))