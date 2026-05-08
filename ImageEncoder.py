import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.encoder_dim = 512 # ResNet18 final channels

    def forward(self, x):
        f = self.cnn(x) # (B, 512, H, W)
        B, C, H, W = f.shape
        return f.view(B, C, -1).permute(0, 2, 1) # (B, P, 512)  P = H*W