import torch
import torch.nn as nn
from torchvision.models import convnext_base

class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        pretrained = None if configs["model"]["pretrained"] == "None" else configs["model"]["pretrained"]

        self.convnext = convnext_base(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.convnext.features.children())[:6])
        
        for param in self.convnext.parameters():
            param.requires_grad = False

        
        self.extra_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(512, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.GELU()
        )

    def forward(self, images: torch.Tensor):
        images = images.permute(1, 0, 2, 3, 4).contiguous()
        feature_maps = []

        for img in torch.split(images, 1, dim=0):
            img = img.squeeze(0)
            features = self.backbone(img)
            features = self.extra_layer(features)
            feature_maps.append(features)

        return torch.stack(feature_maps).permute(1, 0, 2, 3, 4).contiguous()
