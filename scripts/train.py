# scripts/train.py
import os
import json
import torch
import cv2
import numpy as np
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.cuda.amp import autocast, GradScaler  # For mixed precision
import albumentations as A  # For data augmentation
from albumentations.pytorch import ToTensorV2
from sam2.training.trainer import Trainer as SAMTrainer  # Adjust based on actual import from repo

class CustomDataset(Dataset):
    def __init__(self, img_dir: str, ann_dir: str, augment: bool = True):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.ann_files = [f for f in os.listdir(ann_dir) if f.endswith('_ann.json')]
        self.augment = augment

        # Data augmentation pipeline (for training only)
        if self.augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.Transpose(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussianBlur(p=0.3),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))  # Adjust if bboxes are used
        else:
            self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.ann_files)

    def __getitem__(self, idx) -> Dict:
        ann_path = os.path.join(self.ann_dir, self.ann_files[idx])
        with open(ann_path, 'r') as f:
            ann = json.load(f)
        
        img_file = self.ann_files[idx].replace('_ann.json', '.jpg')  # Assume jpg
        img_path = os.path.join(self.img_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency
        
        # Extract masks (assuming ann is list of dicts with 'segmentation' as binary masks or run-length)
        masks = [self._decode_mask(m['segmentation']) for m in ann]  # Implement _decode_mask if needed
        
        # Apply transformations
        transformed = self.transform(image=image, masks=masks)
        image = transformed['image']  # Already a tensor
        masks = transformed['masks']  # List of transformed masks
        
        # Stack masks if needed (depending on SAM2 input format)
        masks = torch.stack([torch.from_numpy(mask).float() for mask in masks]) if masks else torch.empty(0)
        
        return {'image': image, 'masks': masks}

    def _decode_mask(self, segmentation):
        # Implement decoding if segmentation is RLE or other compressed format
        # For example, if it's a list of points, convert to mask
        # Placeholder: assume it's already a numpy array mask
        return np.array(segmentation)  # Adjust as per your annotation format

class OptimizedTrainer:
    def __init__(self, config_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.config_path = config_path
        self.device = device
        # Load config and build trainer from SAM2
        self.trainer = SAMTrainer(config_path)  # Adapt from repo train.py
        self.trainer.to(self.device)  # Move to GPU
        self.scaler = GradScaler()  # For mixed precision

    def train(self, train_dataset: Dataset, val_dataset: Dataset = None, batch_size: int = 8, epochs: int = 20, num_workers: int = 4, lr: float = 1e-4):
        # Use larger batch size if GPU memory allows
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        # Optimizer with learning rate (adjust if trainer has its own)
        optimizer = torch.optim.AdamW(self.trainer.parameters(), lr=lr)  # Example, adapt if needed
        
        for epoch in range(epochs):
            self.trainer.train()  # Set to train mode
            total_loss = 0
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}  # Move to device
                with autocast():  # Mixed precision for speed
                    loss = self.trainer.step(batch)  # Assume step returns loss
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")
            
            # Validation if provided
            if val_dataset:
                self.trainer.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        with autocast():
                            loss = self.trainer.step(batch, train=False)  # Assume step can validate
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss:.4f}")
        
        self.trainer.save_model("checkpoints/finetuned.pt")
        print("[INFO] Training completed and model saved.")

if __name__ == "__main__":
    full_dataset = CustomDataset(
        img_dir="data/processed",
        ann_dir="data/annotated",
        augment=True  # Enable augmentation for training
    )
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # For val, disable augmentation
    val_dataset.dataset.augment = False  # Temporarily disable for val
    
    trainer = OptimizedTrainer(config_path="configs/sam2.1/sam2.1_hiera_b+_MOSE_finetune.yaml")
    trainer.train(train_dataset, val_dataset=val_dataset, batch_size=8, epochs=20, num_workers=4)
