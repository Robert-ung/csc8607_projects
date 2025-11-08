"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

def build_model(config: dict):
    """Construit et retourne un nn.Module selon la config. À implémenter."""
    raise NotImplementedError("build_model doit être implémentée par l'étudiant·e.")

import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection avec projection si nécessaire
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class CNN1DResNet(nn.Module):
    def __init__(self, in_channels=9, num_classes=6, kernel_size=3, num_blocks=(1,1,1)):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.stage1 = self._make_stage(64, 64, num_blocks[0], kernel_size, stride=1)
        self.stage2 = self._make_stage(64, 128, num_blocks[1], kernel_size, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks[2], kernel_size, stride=2)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_stage(self, in_channels, out_channels, num_blocks, kernel_size, stride):
        blocks = []
        blocks.append(ResidualBlock1D(in_channels, out_channels, kernel_size, stride))
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock1D(out_channels, out_channels, kernel_size))
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Global pooling and classifier
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x

def build_model(config: dict) -> torch.nn.Module:
    """Construit le modèle selon la configuration.
    
    Args:
        config (dict): Configuration contenant les paramètres du modèle
        
    Returns:
        torch.nn.Module: Instance du modèle CNN1D résiduel
    """
    kernel_size = config["model"].get("kernel_size", 3)
    num_blocks = config["model"].get("num_blocks", (1,1,1))
    num_classes = config["model"].get("num_classes", 6)
    in_channels = config["model"].get("in_channels", 9)
    
    model = CNN1DResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        kernel_size=kernel_size,
        num_blocks=num_blocks
    )
    return model