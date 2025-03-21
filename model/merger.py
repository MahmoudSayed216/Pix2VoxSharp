import torch
import torch.nn as nn


class Merger(nn.Module):
    def __init__(self, configs):
        super(Merger, self).__init__()
        negative_slop = configs["model"]["negative_slope"]
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=13, out_channels=24, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=24),
            nn.LeakyReLU(negative_slope=negative_slop)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=32),
            nn.LeakyReLU(negative_slope=negative_slop)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=16),
            nn.LeakyReLU(negative_slope=negative_slop)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=8),
            nn.LeakyReLU(negative_slope=negative_slop)
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=1),
            nn.LeakyReLU(negative_slope=negative_slop)
        )


    def forward(self, raw:torch.Tensor, gen:torch.Tensor):
        n_views = raw.size(1)
        raw_features = torch.split(raw, split_size_or_sections=1, dim=1)
        volume_weights = []

        for i in range(n_views):
            raw_feature = raw_features[i].squeeze(dim=1)

            volume_weight = self.conv1(raw_feature)
            volume_weight = self.conv2(volume_weight)
            volume_weight = self.conv3(volume_weight)
            volume_weight = self.conv4(volume_weight)
            volume_weight = self.conv5(volume_weight)

            volume_weight = volume_weight.squeeze(dim=1)

            volume_weights.append(volume_weight)
        volume_weights = torch.stack(volume_weights).permute(1, 0, 2, 3, 4).contiguous()
        volume_weights = torch.softmax(volume_weights, dim=1)

        gen = gen * volume_weights
        gen = torch.sum(gen, dim=1)

        return torch.clamp(gen, min=0, max=1)


