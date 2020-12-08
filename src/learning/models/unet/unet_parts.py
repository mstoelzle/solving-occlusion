"""
Parts of the U-Net model
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, nn_module=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if nn_module is None:
            nn_module = nn

        self.double_conv = nn_module.Sequential(
            nn_module.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn_module.BatchNorm2d(mid_channels),
            nn_module.ReLU(inplace=True),
            nn_module.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn_module.BatchNorm2d(out_channels),
            nn_module.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, nn_module=None):
        super().__init__()

        if nn_module is None:
            nn_module = nn

        self.maxpool_conv = nn_module.Sequential(
            nn_module.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, nn_module=nn_module)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, nn_module=None):
        super().__init__()

        if nn_module is None:
            nn_module = nn

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn_module.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, nn_module=nn_module)
        else:
            self.up = nn_module.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, nn_module=nn_module)

    def forward(self, x1, x2):
        # upsampling
        x1 = self.up(x1)

        if type(x1) == torch.Tensor and type(x2) == torch.Tensor:
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
        elif type(x1) == tuple and type(x2) == tuple:
            # TODO: implement Upsampling for ADF
            raise NotImplementedError

            # adjustments for AFD: we get a mean and a variance as an input
            assert len(x1) == len(x2) == 2
            mu1, var1 = x1
            mu2, var2 = x2

            # input is CHW
            diffY = mu2.size(2) - mu1.size(2)
            diffX = mu2.size(3) - mu1.size(3)

            padding_config = [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]

            mu1 = F.pad(mu1, padding_config)
            var1 = F.pad(var1, padding_config)

            mu = torch.cat([mu2, mu1], dim=1)
            var = torch.cat([var2, var1], dim=1)

            return self.conv(mu, var)
        else:
            raise NotImplementedError


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, nn_module=None):
        super(OutConv, self).__init__()

        if nn_module is None:
            nn_module = nn

        self.conv = nn_module.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
