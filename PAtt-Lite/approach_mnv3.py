import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
from tqdm import tqdm
import os

# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=1)

    def forward(self, x):
        x = x.unsqueeze(0)
        x, _ = self.attention(x, x, x)
        return x.squeeze(0)
# Custom Model
class CustomModel(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(CustomModel, self).__init__()
        
        # Base Model (MobileNetV3 Backbone)
        self.base_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.base_model.features = nn.Sequential(*list(self.base_model.features.children())[:-1])
        
        # Patch Extraction
        self.patch_extraction = nn.Sequential(
            nn.Conv2d(160, 256, kernel_size=4, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        
        # Pooling and Pre-Classification Layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pre_classification = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        # Attention Layer
        self.attention = AttentionBlock(d_model=32)
        
        # Classification Head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.patch_extraction(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.pre_classification(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
