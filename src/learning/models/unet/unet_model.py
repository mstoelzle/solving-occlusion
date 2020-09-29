""" Full assembly of the parts to form the complete network """

from torch import nn
from typing import *

from .unet_parts import *
from ..base_model import BaseModel
from src.enums import *


class UNet(BaseModel):
    def __init__(self, hidden_dims: List = None, bilinear=True, **kwargs):
        super(UNet, self).__init__(**kwargs)

        self.bilinear = bilinear
        self.hidden_dims = hidden_dims
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256, 512, 1024]

        factor = 2 if bilinear else 1

        encoder_layers = [DoubleConv(len(self.in_channels), self.hidden_dims[0])]
        for in_idx, num_out_channels in enumerate(self.hidden_dims[1:]):
            if (in_idx + 1) >= len(self.hidden_dims[1:]):
                encoder_layers.append(Down(self.hidden_dims[in_idx], num_out_channels // factor))
            else:
                encoder_layers.append(Down(self.hidden_dims[in_idx], num_out_channels))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        reversed_hidden_dims = self.hidden_dims.copy()
        reversed_hidden_dims.reverse()
        for in_idx, num_out_channels in enumerate(reversed_hidden_dims[1:]):
            if (in_idx + 1) >= len(reversed_hidden_dims[1:]):
                decoder_layers.append(Up(reversed_hidden_dims[in_idx], num_out_channels, self.bilinear))
            else:
                decoder_layers.append(Up(reversed_hidden_dims[in_idx], num_out_channels // factor, self.bilinear))
        decoder_layers.append(OutConv(reversed_hidden_dims[-1], len(self.out_channels)))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, data: Dict[Union[str, ChannelEnum], torch.Tensor],
                **kwargs) -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        input, norm_consts = self.assemble_input(data)

        encodings = []
        for encoding_idx, encoder_layer in enumerate(self.encoder):
            if len(encodings) == 0:
                encodings.append(encoder_layer(input))
            else:
                encodings.append(encoder_layer(encodings[-1]))

        encodings.reverse()

        x = encodings[0]
        for decoding_idx, decoder_layer in enumerate(self.decoder):
            if decoding_idx + 1 < len(self.decoder):
                x = decoder_layer(x, encodings[decoding_idx+1])
            else:
                x = decoder_layer(x)

        output = {ChannelEnum.RECONSTRUCTED_ELEVATION_MAP: x.squeeze()}

        output = self.denormalize_output(output, norm_consts)

        return output

    def loss_function(self,
                      loss_config: dict,
                      output: Dict[Union[ChannelEnum, LossEnum, str], torch.Tensor],
                      data: Dict[ChannelEnum, torch.Tensor],
                      **kwargs) -> dict:

        loss_dict = self.eval_loss_function(loss_config=loss_config, output=output, data=data, **kwargs)

        if self.training:
            weights = loss_config.get("train_weights", {})

            reconstruction_weight = weights.get(LossEnum.RECONSTRUCTION.value, 1)
            reconstruction_occlusion_weight = weights.get(LossEnum.RECONSTRUCTION_OCCLUSION.value, 1)

            loss = reconstruction_weight * loss_dict[LossEnum.RECONSTRUCTION] \
                   + reconstruction_occlusion_weight * loss_dict[LossEnum.RECONSTRUCTION_OCCLUSION]

            loss_dict.update({LossEnum.LOSS: loss})

            return loss_dict
        else:
            return loss_dict
