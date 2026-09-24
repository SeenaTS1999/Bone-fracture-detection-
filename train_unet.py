import os
import cv2
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import UNetEnhancer

# Optimize CPU multi-threading
torch.set_num_threads(os.cpu_count() or 4)

class FastUNetDataset(Dataset):
    def __init__(self, hq_dir, lq_dir, max_samples=1000):
        self.data = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        
        # 1. Map all HQ image paths
        hq_map = {}
        for root, _, files in os.walk(hq_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    hq_map[f] = os.path.join(root, f)
                    
        # 2. Collect matching pairs
        matched_pairs = []
        for root, _, files in os.walk(lq_dir):
            for f in files:
                if f.lower().endswith(valid_exts) and f in hq_map:
                    matched_pairs.append((os.path.join(root, f), hq_map[f]))

        # Limit to a high-diversity subset (saves hours while maintaining quality)
        if len(matched_pairs) > max_samples:
            random.seed(42)
            matched_pairs = random.sample(matched_pairs, max_samples)

        print(f"Pre-loading {len(matched_pairs)} images directly into RAM for maximum speed...")
        
        # 3. Cache directly into memory to remove disk read delays
        for lq_path, hq_path in matched_pairs:
            lq = cv2.imread(lq_path, cv2.IMREAD_GRAYSCALE)
            hq = cv2.imread(hq_path, cv2.IMREAD_GRAYSCALE)
            
            if lq is not None and hq is not None:
                lq = cv2.resize(lq, (256, 256)).astype('float32') / 255.0
                hq = cv2.resize(hq, (256, 256)).astype('float32') / 255.0
                
                lq_tensor = torch.from_numpy(lq).unsqueeze(0)
                hq_tensor = torch.from_numpy(hq).unsqueeze(0)
                self.data.append((lq_tensor, hq_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running optimized training on: {device}")
    
    # 1000 balanced pairs is ideal for synthetic degradation tasks
    dataset = FastUNetDataset(hq_dir="archive (5)", lq_dir="dataset_lq", max_samples=1000)
    
    if len(dataset) == 0:
        print("[Error] No paired images loaded. Verify paths.")
        return
        
    # Larger batch size (16) optimizes CPU vectorization instructions
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = UNetEnhancer().to(device)
    criterion = nn.L1Loss()  # Fidelity-aware L1 reconstruction loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 5
    print("\nTraining starting...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for step, (lq_imgs, hq_imgs) in enumerate(dataloader):
            lq_imgs, hq_imgs = lq_imgs.to(device), hq_imgs.to(device)
            
            optimizer.zero_grad(set_to_none=True)  # Faster memory clean
            outputs = model(lq_imgs)
            loss = criterion(outputs, hq_imgs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Completed - Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), "unet_enhancer.pth")
    print("\nSaved high-quality weights to unet_enhancer.pth!")

if __name__ == "__main__":
    train_unet()