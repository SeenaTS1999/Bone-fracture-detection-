import os
import cv2
import numpy as np

def degrade_image(img):
    # Resize to the standard 256x256 network dimension
    img_resized = cv2.resize(img, (256, 256))
    
    # 1. Very faint blur (sigmaX=0.4 applies barely noticeable smoothing)
    blurred = cv2.GaussianBlur(img_resized, (3, 3), sigmaX=0.4)
    
    # 2. Minimal sensor noise (scale=3 introduces just a hint of sensor grain)
    noise = np.random.normal(0, 3, blurred.shape).astype(np.float32)
    noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 3. Very slight contrast reduction (alpha=0.95 keeps 95% of original contrast)
    minimal_lq = cv2.convertScaleAbs(noisy, alpha=0.95, beta=2)
    
    return minimal_lq

def process_dataset(source_root, target_root):
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    count = 0

    if not os.path.exists(source_root):
        print(f"Error: Folder '{source_root}' does not exist.")
        return

    print(f"Scanning '{source_root}' for images...")

    for root, _, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(valid_exts):
                src_file = os.path.join(root, file)
                
                # Retain folder structure
                rel_path = os.path.relpath(root, source_root)
                dest_dir = os.path.join(target_root, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                dest_file = os.path.join(dest_dir, file)

                img = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    lq_img = degrade_image(img)
                    cv2.imwrite(dest_file, lq_img)
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} images...")

    if count == 0:
        print("\n[Alert] 0 images found. Check your folder path.")
    else:
        print(f"\nSuccess! Processed {count} images saved to: {target_root}")

if __name__ == "__main__":
    source_folder = "archive (5)"
    destination_folder = "dataset_lq"
    
    process_dataset(source_folder, destination_folder)