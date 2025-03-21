import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()

        negative_slop = configs["model"]["negative_slope"]
        bias = configs["model"]["use_bias"]

        self.upsample1 = nn.Sequential(nn.ConvTranspose3d(in_channels=3072, out_channels=768, kernel_size=4, stride=2, padding=1, bias=bias),
                                       nn.BatchNorm3d(num_features=768),
                                       nn.LeakyReLU(negative_slop)) 
        
        self.upsample2 = nn.Sequential(nn.ConvTranspose3d(in_channels=768, out_channels=192, kernel_size=4, stride=2, padding=1, bias=bias),
                                       nn.BatchNorm3d(num_features=192),
                                       nn.LeakyReLU(negative_slop))
        
        self.upsample3 = nn.Sequential(nn.ConvTranspose3d(in_channels=192, out_channels=48, kernel_size=4, stride=2, padding=1, bias=bias),
                                       nn.BatchNorm3d(num_features=48),
                                       nn.LeakyReLU(negative_slop)
                                       )
        
        self.upsample4 = nn.Sequential(nn.ConvTranspose3d(in_channels=48, out_channels=12, kernel_size=4, stride=2, padding=1, bias=bias),
                                       nn.BatchNorm3d(num_features=12),
                                       nn.LeakyReLU(negative_slop)
                                       )
        
        self.upsample5 = nn.Sequential(nn.ConvTranspose3d(in_channels=12, out_channels=1, kernel_size=1, bias=bias))


    def forward(self, features_maps_list):
        gen_volumes = []
        raw_features = []
        
        features_maps_list = features_maps_list.permute(1, 0, 2, 3, 4).contiguous()
        features_maps_list = torch.split(features_maps_list, 1, dim=0)

        for feature_maps in features_maps_list:
            volume = feature_maps.view(-1, 3072, 2, 2, 2)
            volume = self.upsample1(volume)
            volume = self.upsample2(volume)
            volume = self.upsample3(volume)
            volume = self.upsample4(volume)
            raw_feature = volume
            volume = self.upsample5(volume)
            raw_feature = torch.cat((raw_feature, volume), dim=1)

            gen_volumes.append(torch.squeeze(volume, dim=1))
            raw_features.append(raw_feature)

        gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        return raw_features, gen_volumes

