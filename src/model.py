import torch
import torch.nn as nn
from torchvision import models
class ResNet18Baseline(nn.Module):
    """
    Baseline model:
    ResNet18 + Binary Classification Head
    Input:
        image -> [B, 3, 224, 224]
    Output:
        logits -> [B]
    """
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        # Load ResNet18
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Number of features before original FC layer
        in_features = self.backbone.fc.in_features

        # Replace original ImageNet classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )
    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        return:
            logits: [B]
        """
        logits = self.backbone(x)
        return logits.squeeze(1)
class SiameseResNet18(nn.Module):
    """
    Siamese ResNet18
    Three inputs:
        anchor
        positive
        negative
    All three pass through the SAME ResNet18 backbone.
    Output:
        anchor_embedding
        positive_embedding
        negative_embedding
    """
    def __init__(
        self,
        pretrained=True,
        embedding_dim=128,
        dropout=0.3
    ):
        super().__init__()

        # Shared ResNet18 backbone
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Get feature dimension
        in_features = self.backbone.fc.in_features
        # Remove original ImageNet classifier
        self.backbone.fc = nn.Identity()
        # Projection head
        # ResNet18 feature -> embedding
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def encode(self, x):
        """
        Convert an image into an embedding.
        Input:
            x: [B, 3, 224, 224]
        Output:
            embedding: [B, embedding_dim]
        """
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        return embedding
    def forward(self, anchor, positive, negative):
        """
        Forward pass for a triplet.
        anchor:
            [B, 3, 224, 224]
        positive:
            [B, 3, 224, 224]
        negative:
            [B, 3, 224, 224]
        """
        anchor_embedding = self.encode(anchor)
        positive_embedding = self.encode(positive)
        negative_embedding = self.encode(negative)
        return (
            anchor_embedding,
            positive_embedding,
            negative_embedding
        )
# Simple test
if __name__ == "__main__":
    # Create dummy images
    batch_size = 4
    anchor = torch.randn(
        batch_size, 3, 224, 224
    )
    positive = torch.randn(
        batch_size, 3, 224, 224
    )
    negative = torch.randn(
        batch_size, 3, 224, 224
    )
    # Test Baseline
    baseline = ResNet18Baseline(
        pretrained=False
    )
    baseline_output = baseline(anchor)
    print("=== Baseline ===")
    print("Input shape:", anchor.shape)
    print("Output shape:", baseline_output.shape)
    # Test Siamese
    siamese = SiameseResNet18(
        pretrained=False,
        embedding_dim=128
    )
    a_emb, p_emb, n_emb = siamese(
        anchor,
        positive,
        negative
    )
    print("\n=== Siamese ResNet18 ===")
    print("Anchor embedding:", a_emb.shape)
    print("Positive embedding:", p_emb.shape)
    print("Negative embedding:", n_emb.shape)