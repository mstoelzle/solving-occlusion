""" Full assembly of the parts to form the complete network """

from typing import *

from src.dataloaders.dataloader_meta_info import DataloaderMetaInfo
from src.enums import *
from src.learning.loss.loss import masked_total_variation_loss_fct, adf_heteroscedastic_loss_fct
from .unet_parts import *
from ..base_model import BaseModel


class UNet(BaseModel):
    def __init__(self, hidden_dims: List = None, bilinear=True, **kwargs):
        super(UNet, self).__init__(**kwargs)

        nn_module = adf if self.adf else nn

        self.bilinear = bilinear
        self.hidden_dims = hidden_dims
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256, 512, 1024]

        # number of convolutions per layer (in the original paper two convolutions per layer)
        self.num_conv_per_layer = self.config.get("num_conv_per_layer", 2)

        factor = 2 if bilinear else 1

        encoder_layers = []
        for in_idx, num_out_channels in enumerate(self.hidden_dims):
            if in_idx + 1 >= len(self.hidden_dims):
                num_down_out_channels = num_out_channels // factor
            else:
                num_down_out_channels = num_out_channels

            if type(self.dropout_p) == list:
                layer_dropout_p = self.dropout_p[in_idx]
            else:
                layer_dropout_p = self.dropout_p

            if in_idx == 0:
                encoder_layers.append(DoubleConv(len(self.in_channels), self.hidden_dims[in_idx],
                                                 num_conv_per_layer=self.num_conv_per_layer,
                                                 nn_module=nn_module, dropout_p=layer_dropout_p,
                                                 keep_variance_fn=self.keep_variance_fn))
            else:
                encoder_layers.append(Down(self.hidden_dims[in_idx-1], num_down_out_channels,
                                           num_conv_per_layer=self.num_conv_per_layer,
                                           nn_module=nn_module, dropout_p=layer_dropout_p,
                                           keep_variance_fn=self.keep_variance_fn))

        self.encoder = nn_module.Sequential(*encoder_layers)

        decoder_layers = []
        reversed_hidden_dims = self.hidden_dims.copy()
        reversed_hidden_dims.reverse()

        for in_idx, num_out_channels in enumerate(reversed_hidden_dims[1:]):
            if (in_idx + 1) >= len(reversed_hidden_dims[1:]):
                num_up_out_channels = num_out_channels
            else:
                num_up_out_channels = num_out_channels // factor

            if type(self.dropout_p) == list:
                layer_dropout_p = self.dropout_p[-(1+in_idx)]
            else:
                layer_dropout_p = self.dropout_p

            decoder_layers.append(Up(reversed_hidden_dims[in_idx], num_up_out_channels,
                                     self.bilinear, num_conv_per_layer=self.num_conv_per_layer,
                                     nn_module=nn_module, dropout_p=layer_dropout_p,
                                     keep_variance_fn=self.keep_variance_fn))

        decoder_layers.append(OutConv(reversed_hidden_dims[-1], len(self.out_channels),
                                      nn_module=nn_module, keep_variance_fn=self.keep_variance_fn))

        self.decoder = nn_module.Sequential(*decoder_layers)

        self.feature_extractor = None

    def forward(self, input: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.adf:
            encodings = []
            for encoding_idx, encoder_layer in enumerate(self.encoder):
                if len(encodings) == 0:
                    encodings.append(encoder_layer(*input))
                else:
                    encodings.append(encoder_layer(*encodings[-1]))

            encodings.reverse()

            x = encodings[0]
            for decoding_idx, decoder_layer in enumerate(self.decoder):
                if isinstance(decoder_layer, Up):
                    x = decoder_layer(x, encodings[decoding_idx + 1])
                else:
                    x = decoder_layer(*x)
        else:
            encodings = []
            for encoding_idx, encoder_layer in enumerate(self.encoder):
                if len(encodings) == 0:
                    encodings.append(encoder_layer(input))
                else:
                    encodings.append(encoder_layer(encodings[-1]))

            encodings.reverse()

            x = encodings[0]
            for decoding_idx, decoder_layer in enumerate(self.decoder):
                if decoding_idx < len(self.decoder) - 1:
                    x = decoder_layer(x, encodings[decoding_idx + 1])
                else:
                    x = decoder_layer(x)

        return x

    def loss_function(self,
                      loss_config: dict,
                      output: Dict[Union[ChannelEnum, LossEnum, str], torch.Tensor],
                      data: Dict[ChannelEnum, torch.Tensor],
                      dataloader_meta_info: DataloaderMetaInfo = None,
                      **kwargs) -> dict:

        loss_dict = self.eval_loss_function(loss_config=loss_config, output=output, data=data,
                                            dataloader_meta_info=dataloader_meta_info, **kwargs)

        if self.training:
            weights = loss_config.get("train_weights", {})

            reconstruction_weight = weights.get(LossEnum.MSE_REC_ALL.value, 0)
            reconstruction_non_occlusion_weight = weights.get(LossEnum.MSE_REC_NOCC.value, 1)
            reconstruction_occlusion_weight = weights.get(LossEnum.MSE_REC_OCC.value, 1)
            perceptual_weight = weights.get(LossEnum.PERCEPTUAL.value, 0)
            style_weight = weights.get(LossEnum.STYLE.value, 0)
            total_variation_weight = weights.get(LossEnum.TV.value, 0)
            adf_het_weight = weights.get(LossEnum.ADF_HET.value, 0)
            mse_rec_data_um_nocc_weight = weights.get(LossEnum.MSE_REC_DATA_UM_NOCC.value, 0)

            if perceptual_weight > 0 or style_weight > 0:
                artistic_loss = self.artistic_loss_function(loss_config=loss_config, output=output, data=data, **kwargs)
                loss_dict.update(artistic_loss)

            if total_variation_weight > 0:
                loss_dict[LossEnum.TV] = masked_total_variation_loss_fct(input=output[ChannelEnum.COMP_DEM],
                                                                         mask=data[ChannelEnum.OCC_MASK])

            if self.adf and adf_het_weight > 0:
                loss_dict[LossEnum.ADF_HET] = adf_heteroscedastic_loss_fct(mu=output[ChannelEnum.REC_DEM],
                                                                           log_var=output[ChannelEnum.REC_DATA_UM],
                                                                           target=data[ChannelEnum.GT_DEM])

            loss = reconstruction_weight * loss_dict[LossEnum.MSE_REC_ALL] \
                   + reconstruction_non_occlusion_weight * loss_dict[LossEnum.MSE_REC_NOCC] \
                   + reconstruction_occlusion_weight * loss_dict[LossEnum.MSE_REC_OCC] \
                   + mse_rec_data_um_nocc_weight * loss_dict.get(LossEnum.MSE_REC_DATA_UM_NOCC, 0.)

            if perceptual_weight > 0:
                loss += perceptual_weight * loss_dict.get(LossEnum.PERCEPTUAL, 0.)

            if style_weight > 0:
                loss += style_weight * loss_dict.get(LossEnum.STYLE, 0.)

            if total_variation_weight > 0:
                loss += total_variation_weight * loss_dict.get(LossEnum.TV, 0.)

            if self.adf and adf_het_weight > 0:
                loss += adf_het_weight * loss_dict.get(LossEnum.ADF_HET, 0.)

            loss_dict.update({LossEnum.LOSS: loss})

            return loss_dict
        else:
            return loss_dict
