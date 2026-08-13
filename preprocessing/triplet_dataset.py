import os
import pandas as pd
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class DeepfakeTripletDataset(Dataset):
    def __init__(self, csv_file, split='train', transform=None):
        
        # Đọc file CSV
        self.full_df = pd.read_csv(csv_file)
        
        # Lọc data theo split (train, val, hoặc test)
        self.df = self.full_df[self.full_df['split'] == split].reset_index(drop=True)
        self.transform = transform
        
        self.video_groups = self.df.groupby('video_id')
        
        self.real_df = self.df[self.df['label'] == 'real']
        self.fake_df = self.df[self.df['label'] == 'fake']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #chọn anchor
        anchor_row = self.df.iloc[idx]
        anchor_img_path = anchor_row['image_path']
        anchor_label = anchor_row['label']
        anchor_video_id = anchor_row['video_id']

        # chọn positive
        pos_candidates = self.video_groups.get_group(anchor_video_id)
        if len(pos_candidates) > 1:
          
            pos_candidates = pos_candidates[pos_candidates.index != idx]
        
        pos_row = pos_candidates.sample(n=1).iloc[0]
        pos_img_path = pos_row['image_path']

        # chọn negative
        if anchor_label == 'real':
            # Nếu Anchor là Real -> Negative phải là Fake
            neg_row = self.fake_df.sample(n=1).iloc[0]
        else:
            # Nếu Anchor là Fake -> Negative phải là Real
            neg_row = self.real_df.sample(n=1).iloc[0]
            
        neg_img_path = neg_row['image_path']

        
        anchor_img = Image.open(anchor_img_path).convert('RGB')
        pos_img = Image.open(pos_img_path).convert('RGB')
        neg_img = Image.open(neg_img_path).convert('RGB')

      
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img

def imshow_triplet(anchor, pos, neg):
   
    def unnormalize(tensor):
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = tensor.numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        return img

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(unnormalize(anchor))
    axes[0].set_title("Anchor")
    axes[0].axis('off')
    
    axes[1].imshow(unnormalize(pos))
    axes[1].set_title("Positive (Cùng video)")
    axes[1].axis('off')
    
    axes[2].imshow(unnormalize(neg))
    axes[2].set_title("Negative (Khác class)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    CSV_PATH = r"D:\python\output_dataset\metadata.csv"
    
    # Định nghĩa Normalize Pipeline (Chuẩn hóa)
    # Resize, đưa về Tensor (0-1), và Normalize theo ImageNet
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = DeepfakeTripletDataset(csv_file=CSV_PATH, split='train', transform=data_transforms)
    
    print(f"Tổng số mẫu trong tập Train: {len(train_dataset)}")
    
    
    anchor, positive, negative = train_dataset[0]
    
    print("Kích thước Tensor Anchor:", anchor.shape)
    
    print("Đang vẽ biểu đồ kiểm tra bộ 3 (Triplet)... ")
    imshow_triplet(anchor, positive, negative)