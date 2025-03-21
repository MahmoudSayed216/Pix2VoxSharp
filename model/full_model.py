import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .merger import Merger
from .refiner import Refiner


class Pix2VoxSharp(nn.Module):
    def __init__(self, configs):
        super(Pix2VoxSharp, self).__init__()
        self.set_merger(False)
        self.encoder = Encoder(configs)
        self.decoder = Decoder(configs)
        self.merger = Merger(configs)
        self.refiner = Refiner(configs)

    def forward(self, x):
        feature_maps = self.encoder(x)
        raw, gen = self.decoder(feature_maps)
        if self.USE_MERGER:
            volume = self.merger(raw, gen)
            volume = self.refiner(volume)
        else:
            volume = gen.squeeze(dim=1)
            
        return volume

    def set_merger(self, merger_state: bool)-> None:
         self.USE_MERGER = merger_state