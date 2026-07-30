""" Full assembly of the parts to form the complete network """

import torch
from .unet_parts import *


class CheckpointWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        def run(*inner_args, **inner_kwargs):
            return self.module(*inner_args, **inner_kwargs)

        return torch.utils.checkpoint(run, *args, use_reentrant=False, **kwargs)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = CheckpointWrapper(self.inc)
        self.down1 = CheckpointWrapper(self.down1)
        self.down2 = CheckpointWrapper(self.down2)
        self.down3 = CheckpointWrapper(self.down3)
        self.down4 = CheckpointWrapper(self.down4)
        self.up1 = CheckpointWrapper(self.up1)
        self.up2 = CheckpointWrapper(self.up2)
        self.up3 = CheckpointWrapper(self.up3)
        self.up4 = CheckpointWrapper(self.up4)
        self.outc = CheckpointWrapper(self.outc)
