import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Baseline(nn.Module): 
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1) 
        ) 
    def forward(self, x):
        logits = self.backbone(x)
        return logits.squeeze(1)