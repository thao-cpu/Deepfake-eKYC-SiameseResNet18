import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SiameseResNet18(nn.Module):
    # Đổi dropout mặc định thành 0.4 cho khớp bản train mới nhất
    def __init__(self, pretrained=False, embedding_dim=128, dropout=0.4):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # BẢN MỚI CÓ THÊM DROPOUT Ở CLASSIFIER
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )

    def encode(self, x):
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x):
        emb = self.encode(x)
        # Chỉ trả về logits để khớp với hàm sigmoid trong inference.py
        logits = self.classifier(emb).squeeze(1)
        return logits