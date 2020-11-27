import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import *

from ..base_model import BaseModel
from src.enums import *
from src.datasets.base_dataset import BaseDataset
from src.learning.loss.loss import total_variation_loss_fct, masked_total_variation_loss_fct


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
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


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False, partial_conv: bool = True):
        super().__init__()

        if sample == 'down-7':
            kernel_size = 7
            stride = 2
            padding = 3
        elif sample == 'down-5':
            kernel_size = 5
            stride = 2
            padding = 2
        elif sample == 'down-3':
            kernel_size = 3
            stride = 2
            padding = 1
        else:
            kernel_size = 3
            stride = 1
            padding = 1

        if partial_conv:
            self.conv = PartialConv(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=conv_bias)
        else:
            # this uses just vanilla convolutions instead of partial convolutions
            self.conv = nn.Conv2d(2*in_ch, 2*out_ch, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        if isinstance(self.conv, PartialConv):
            h, h_mask = self.conv(input, input_mask)
        else:
            # this uses just vanilla convolutions instead of partial convolutions
            conv_output = self.conv(torch.cat((input, input_mask), dim=1))
            num_channels = conv_output.size(1)
            h = conv_output[:, 0:int(num_channels // 2), ...]
            h_mask = conv_output[:, int(num_channels // 2):, ...]

        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class PartialConvUNet(BaseModel):
    # https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
    def __init__(self, hidden_dims: List = None, num_layers=None, upsampling_mode='nearest', partial_conv: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

        # we dont have RGB images but rather 1-channel inputs. The mask channel is accounted for separately
        input_channels = len(self.in_channels) - 1

        # this only works for input channels occluded elevation map and binary occlusion map
        assert self.in_channels == [ChannelEnum.OCCLUDED_ELEVATION_MAP, ChannelEnum.BINARY_OCCLUSION_MAP]

        # either we use the standard Nvidia architecture or our own hidden_dims specification
        assert hidden_dims is None or num_layers is None
        assert hidden_dims is not None or num_layers is not None

        self.partial_conv = partial_conv
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode

        if hidden_dims is None:
            self.num_layers = num_layers

            self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7', partial_conv=self.partial_conv)
            self.enc_2 = PCBActiv(64, 128, sample='down-5', partial_conv=self.partial_conv)
            self.enc_3 = PCBActiv(128, 256, sample='down-5', partial_conv=self.partial_conv)
            self.enc_4 = PCBActiv(256, 512, sample='down-3', partial_conv=self.partial_conv)
            for i in range(4, self.num_layers):
                name = 'enc_{:d}'.format(i + 1)
                setattr(self, name, PCBActiv(512, 512, sample='down-3', partial_conv=self.partial_conv))

            for i in range(4, self.num_layers):
                name = 'dec_{:d}'.format(i + 1)
                setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky', partial_conv=self.partial_conv))
            self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky', partial_conv=self.partial_conv)
            self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky', partial_conv=self.partial_conv)
            self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky', partial_conv=self.partial_conv)
            self.dec_1 = PCBActiv(64 + input_channels, len(self.out_channels),
                                  bn=False, activ=None, conv_bias=True, partial_conv=self.partial_conv)
        else:
            self.num_layers = len(hidden_dims)
            self.hidden_dims = hidden_dims

            in_channels = input_channels
            for i, out_channels in enumerate(self.hidden_dims):
                bn = True
                if i == 0:
                    bn = False

                name = f"enc_{i+1}"
                self.__setattr__(name, PCBActiv(in_channels, out_channels, bn=bn, sample="down-3",
                                                partial_conv=self.partial_conv))

                in_channels = out_channels

            for i, out_channels in enumerate(reversed(self.hidden_dims[:-1])):
                name = f"dec_{len(self.hidden_dims) - i}"
                self.__setattr__(name, PCBActiv(in_channels + out_channels, out_channels, activ='leaky',
                                                partial_conv=self.partial_conv))

                in_channels = out_channels

            self.dec_1 = PCBActiv(in_channels + input_channels, len(self.out_channels),
                                  bn=False, activ=None, conv_bias=True, partial_conv=self.partial_conv)


    def forward(self, data: Dict[Union[str, ChannelEnum], torch.Tensor],
                **kwargs) -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        input, norm_consts = self.assemble_input(data)

        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        # input and mask
        image = input[:, 0:1, ...]
        # image = torch.cat((input[:, 0:1, ...], input[:, 0:1, ...], input[:, 0:1, ...]), dim=1)

        mask = input[:, 1:2, ...]
        # mask = torch.cat((input[:, 1:2, ...], input[:, 1:2, ...], input[:, 1:2, ...]), dim=1)
        h_dict['h_0'], h_mask_dict['h_0'] = image, mask

        h_key_prev = 'h_0'
        for i in range(1, self.num_layers + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.num_layers)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8

        for i in range(self.num_layers, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        output = {ChannelEnum.RECONSTRUCTED_ELEVATION_MAP: h[:, 0, ...]}

        output = self.denormalize_output(data, output, norm_consts)

        return output

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()

    def loss_function(self,
                      loss_config: dict,
                      output: Dict[Union[ChannelEnum, LossEnum, str], torch.Tensor],
                      data: Dict[ChannelEnum, torch.Tensor],
                      dataset: BaseDataset = None,
                      **kwargs) -> dict:

        loss_dict = self.eval_loss_function(loss_config=loss_config, output=output, data=data, dataset=dataset,**kwargs)

        if self.training:
            weights = loss_config.get("train_weights", {})

            reconstruction_non_occlusion_weight = weights.get(LossEnum.MSE_REC_NOCC.value, 1)
            reconstruction_occlusion_weight = weights.get(LossEnum.MSE_REC_OCC.value, 1)
            total_variation_weight = weights.get(LossEnum.TV.value, 0)

            total_variation_loss = masked_total_variation_loss_fct(input=output[ChannelEnum.COMPOSED_ELEVATION_MAP],
                                                                   mask=data[ChannelEnum.BINARY_OCCLUSION_MAP])

            loss = reconstruction_non_occlusion_weight * loss_dict[LossEnum.MSE_REC_NOCC] \
                   + reconstruction_occlusion_weight * loss_dict[LossEnum.MSE_REC_OCC] \
                   + total_variation_weight * total_variation_loss

            loss_dict.update({LossEnum.LOSS: loss})

            return loss_dict
        else:
            return loss_dict
