import torch
import torch.nn as nn

class Refiner(torch.nn.Module):
    def __init__(self, configs):
        super(Refiner, self).__init__()

        negative_slop = configs["model"]["negative_slope"]
        bias = configs["model"]["use_bias"]

        self.conv1 = torch.nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, padding=2),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=negative_slop),
            nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=4, padding=2),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(negative_slope=negative_slop),
            nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=4, padding=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(negative_slope=negative_slop),
            nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            nn.Linear(8192, 2048),
            nn.ReLU()
        )
        self.conv5 = torch.nn.Sequential(
            nn.Linear(2048, 8192),
            nn.ReLU()
        )
        self.conv6 = torch.nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=bias, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.conv7 = torch.nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=bias, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.conv8 = torch.nn.Sequential(
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=bias, padding=1),
            nn.Sigmoid()
        )

    def forward(self, coarse_volumes):
        #downsample
        volumes_32_l = coarse_volumes.view((-1, 1, 32, 32, 32))
        volumes_16_l = self.conv1(volumes_32_l)
        volumes_8_l = self.conv2(volumes_16_l)
        volumes_4_l = self.conv3(volumes_8_l)
        #linear bottleneck
        flatten_features = self.conv4(volumes_4_l.view(-1, 8192))
        flatten_features = self.conv5(flatten_features)
        
        #upsample + skip connections
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, 4, 4, 4)
        volumes_8_r = volumes_8_l + self.conv6(volumes_4_r)
        volumes_16_r = volumes_16_l + self.conv7(volumes_8_r)
        volumes_32_r = (volumes_32_l + self.conv8(volumes_16_r)) * 0.5

        return volumes_32_r.view((-1, 32, 32, 32))
