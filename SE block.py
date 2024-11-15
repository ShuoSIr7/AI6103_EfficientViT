import torch
import torch.nn as nn

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, channel_scale=1/16):
        super().__init__()
        out_channels = max(1, int(in_channels * channel_scale))
        self.add_module('ap', nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('fc1', nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, bias=True))
        self.add_module('act1', nn.ReLU())
        self.add_module('fc2', nn.Conv2d(
            out_channels, in_channels, 1, 1, 0, 1, bias=True))
        self.add_module('act2', nn.Sigmoid())

    def forward(self, x):
        scale = self.ap(x)
        scale = self.fc1(scale)
        scale = self.act1(scale)
        scale = self.fc2(scale)
        scale = self.act2(scale)
        return scale * x
