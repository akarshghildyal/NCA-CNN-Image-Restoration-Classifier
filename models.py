import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralCA(nn.Module):
    """Neural Cellular Automata for image reconstruction"""
    def __init__(self, channel_n=16, fire_rate=0.5):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        
        self.perception = nn.Conv2d(channel_n, channel_n*3, 3, padding=1, bias=False)
        
        self.update = nn.Sequential(
            nn.Conv2d(channel_n*3, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, channel_n, 1, bias=False)
        )
        
        with torch.no_grad():
            nn.init.xavier_uniform_(self.perception.weight, gain=0.5)
            for layer in self.update:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
            self.update[-1].weight.data.zero_()
    
    def forward(self, x, steps=40, mask=None):
        """Run NCA for specified steps, preserving masked regions"""
        device = next(self.parameters()).device  # Get model device
        
        for _ in range(steps):
            perception = self.perception(x)
            dx = self.update(perception)
            
            if self.training:
                fire_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=device) < self.fire_rate).float()
                dx = dx * fire_mask
            
            x = x + dx
            x = torch.clamp(x, 0, 1)
        
        return x

class SimpleCNN(nn.Module):
    """Simple CNN for Fashion-MNIST classification"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def prepare_nca_input(masked_img, channel_n=16):
    """Convert grayscale to multi-channel state for NCA"""
    device = masked_img.device  
    
    if len(masked_img.shape) == 3:
        masked_img = masked_img.unsqueeze(1)
    
    batch_size = masked_img.shape[0]
    h, w = masked_img.shape[2], masked_img.shape[3]
    
    state = torch.zeros(batch_size, channel_n, h, w, device=device)
    state[:, 0:1] = masked_img
    
    return state