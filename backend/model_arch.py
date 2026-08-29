import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SiameseResNet18(nn.Module):
    def __init__(self, pretrained=False, embedding_dim=128, dropout=0.3):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        # Xóa lớp fc cũ, thay bằng Identity
        self.backbone.fc = nn.Identity()

        # Tạo đầu embedding 
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Đầu phân loại
        self.classifier = nn.Linear(embedding_dim, 1)

    def encode(self, x):
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x):
        emb = self.encode(x)
        # Chỉ lấy logits để nhét vào hàm Sigmoid bên file inference
        logits = self.classifier(emb).squeeze(1)
        return logits