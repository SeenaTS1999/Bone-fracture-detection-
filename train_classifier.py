import os
import cv2
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.models import resnet18, ResNet18_Weights

# Optimize CPU threads
torch.set_num_threads(os.cpu_count() or 4)

class FixedYoloClassificationDataset(Dataset):
    def __init__(self, base_dir, split="train", is_train=True):
        self.samples = []
        self.is_train = is_train
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')

        img_dir = os.path.join(base_dir, split, "images")
        lbl_dir = os.path.join(base_dir, split, "labels")

        if not os.path.exists(img_dir):
            print(f"[Error] Directory not found: {img_dir}")
            return

        for img_name in os.listdir(img_dir):
            if img_name.lower().endswith(valid_exts):
                img_path = os.path.join(img_dir, img_name)
                txt_name = os.path.splitext(img_name)[0] + ".txt"
                txt_path = os.path.join(lbl_dir, txt_name)

                # Definitive label rule from check_labels.py:
                # If .txt file exists and size > 0 -> Fractured (0)
                # If .txt file is 0 bytes or missing -> Normal (1)
                if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                    label = 0  # Fractured
                else:
                    label = 1  # Normal

                self.samples.append((img_path, label))

        frac_cnt = sum(1 for s in self.samples if s[1] == 0)
        norm_cnt = sum(1 for s in self.samples if s[1] == 1)
        print(f"[{split.upper()}] Loaded: {frac_cnt} Fractured (0), {norm_cnt} Normal (1)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((256, 256), dtype=np.uint8)

        img = cv2.resize(img, (256, 256))
        tensor = torch.from_numpy(img.astype('float32') / 255.0).unsqueeze(0)

        # Standard ImageNet normalization for 1-channel grayscale
        tensor = (tensor - 0.485) / 0.229

        if self.is_train:
            if random.random() > 0.5:
                tensor = TF.hflip(tensor)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                tensor = TF.rotate(tensor, angle)

        return tensor, torch.tensor(label, dtype=torch.long)

def build_model():
    # Use ResNet18 adapted for 1-channel grayscale X-rays
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Adapt first convolution to take 1-channel (grayscale) instead of 3 (RGB)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Replace last layer for binary classification: 0 (Fractured), 1 (Normal)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def train_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Classifier on: {device}")

    yolo_base = os.path.join("archive (5)", "BoneFractureYolo8")
    if not os.path.exists(yolo_base):
        yolo_base = os.path.join("archive (5)", "bone fracture detection.v4-v4.yolov8")

    train_dataset = FixedYoloClassificationDataset(base_dir=yolo_base, split="train", is_train=True)
    val_dataset = FixedYoloClassificationDataset(base_dir=yolo_base, split="valid", is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False) if len(val_dataset) > 0 else None

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    epochs = 4
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        train_acc = (train_correct / train_total) * 100 if train_total > 0 else 0

        val_acc_str = ""
        if val_loader:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)
            val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0
            val_acc_str = f" | Val Acc: {val_acc:.2f}%"

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%{val_acc_str}")

    torch.save(model.state_dict(), "fracture_cnn.pth")
    print("\nSaved high-accuracy classifier weights to fracture_cnn.pth!")

if __name__ == "__main__":
    train_classifier()