import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet18Student(nn.Module):
    """
    ResNet18-based student model for knowledge distillation from VTB teacher.
    Replaces the SNN student model while maintaining the same interface.
    """
    def __init__(self, attr_num, feature_dim=768, img_size_h=256, img_size_w=128):
        super().__init__()
        self.attr_num = attr_num
        self.feature_dim = feature_dim
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        
        # Load pretrained ResNet18 (ImageNet pretrained)
        # Note: ImageNet uses different normalization, but knowledge distillation will adapt
        try:
            # New PyTorch API (>=0.13)
            resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except AttributeError:
            # Old PyTorch API
            resnet18 = models.resnet18(pretrained=True)
        
        # Remove the final fully connected layer
        # We'll extract features from layer4 (before avgpool)
        self.backbone = nn.Sequential(*list(resnet18.children())[:-2])
        
        # ResNet18 layer4 output is 512 channels
        # We need to project to feature_dim for consistency with teacher
        # Calculate spatial dimensions after ResNet18 (assuming input 256x128)
        # ResNet18 reduces by factor of 32, so 256x128 -> 8x4
        self.spatial_h = img_size_h // 32
        self.spatial_w = img_size_w // 32
        self.num_patches = self.spatial_h * self.spatial_w
        
        # Projection layer to match teacher feature dimension
        self.feature_proj = nn.Linear(512, feature_dim)
        
        # Classification head for attribute prediction
        # Uses global average pooling followed by MLP
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, attr_num)
        )
        
    def forward(self, x):
        """
        Forward pass of ResNet18 student model.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            logits: Classification logits of shape (B, attr_num)
            features: Spatial features of shape (B, num_patches, feature_dim)
        """
        # Extract features from ResNet18 backbone
        # Output shape: (B, 512, H/32, W/32)
        backbone_features = self.backbone(x)  # (B, 512, 8, 4)
        
        # Get classification logits from backbone features
        logits = self.classifier(backbone_features)  # (B, attr_num)
        
        # Prepare spatial features for knowledge distillation
        # Reshape to (B, num_patches, 512)
        B, C, H, W = backbone_features.shape
        features = backbone_features.view(B, C, H * W)  # (B, 512, num_patches)
        features = features.permute(0, 2, 1)    # (B, num_patches, 512)
        
        # Project to match teacher feature dimension
        features = self.feature_proj(features)  # (B, num_patches, feature_dim)
        
        return logits, features

