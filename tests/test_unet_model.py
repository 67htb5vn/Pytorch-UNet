import torch

from unet.unet_model import UNet


def test_use_checkpointing_does_not_raise():
    model = UNet(n_channels=3, n_classes=2)
    model.use_checkpointing()
    assert isinstance(model.inc, torch.nn.Module)
