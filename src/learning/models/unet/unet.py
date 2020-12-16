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

        factor = 2 if bilinear else 1

        encoder_layers = [DoubleConv(len(self.in_channels), self.hidden_dims[0],
                                     nn_module=nn_module, dropout_p=self.dropout_p,
                                     keep_variance_fn=self.keep_variance_fn)]
        for in_idx, num_out_channels in enumerate(self.hidden_dims[1:]):
            if (in_idx + 1) >= len(self.hidden_dims[1:]):
                num_down_out_channels = num_out_channels // factor
            else:
                num_down_out_channels = num_out_channels
            encoder_layers.append(Down(self.hidden_dims[in_idx], num_down_out_channels,
                                       nn_module=nn_module, dropout_p=self.dropout_p,
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
            decoder_layers.append(Up(reversed_hidden_dims[in_idx], num_up_out_channels,
                                     self.bilinear, nn_module=nn_module, dropout_p=self.dropout_p,
                                     keep_variance_fn=self.keep_variance_fn))

        decoder_layers.append(OutConv(reversed_hidden_dims[-1], len(self.out_channels), nn_module=nn_module,
                                      keep_variance_fn=self.keep_variance_fn))

        self.decoder = nn_module.Sequential(*decoder_layers)

        self.feature_extractor = None

    def forward_pass(self, input: torch.Tensor, data: dict, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
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

            # remove channels dimension from tensor
            for i in range(len(x)):
                x[i] = x[i].squeeze(dim=1)
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
                if decoding_idx + 1 < len(self.decoder):
                    x = decoder_layer(x, encodings[decoding_idx + 1])
                else:
                    x = decoder_layer(x)

            x = x.squeeze(dim=1)

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

            if perceptual_weight > 0 or style_weight > 0:
                artistic_loss = self.artistic_loss_function(loss_config=loss_config, output=output, data=data, **kwargs)
                loss_dict.update(artistic_loss)

            if total_variation_weight > 0:
                loss_dict[LossEnum.TV] = masked_total_variation_loss_fct(input=output[ChannelEnum.COMP_DEM],
                                                                         mask=data[ChannelEnum.OCC_MASK])

            if self.adf and adf_het_weight > 0:
                loss_dict[LossEnum.ADF_HET] = adf_heteroscedastic_loss_fct(mu=output[ChannelEnum.REC_DEM],
                                                                           log_var=output[ChannelEnum.DATA_UNCERTAINTY_MAP],
                                                                           target=data[ChannelEnum.GT_DEM])

            loss = reconstruction_weight * loss_dict[LossEnum.MSE_REC_ALL] \
                   + reconstruction_non_occlusion_weight * loss_dict[LossEnum.MSE_REC_NOCC] \
                   + reconstruction_occlusion_weight * loss_dict[LossEnum.MSE_REC_OCC] \
                   + perceptual_weight * loss_dict.get(LossEnum.PERCEPTUAL, 0.) \
                   + style_weight * loss_dict.get(LossEnum.STYLE, 0.) \
                   + total_variation_weight * loss_dict.get(LossEnum.TV, 0.)

            loss_dict.update({LossEnum.LOSS: loss})

            return loss_dict
        else:
            return loss_dict

    def train(self, mode: bool = True, **kwargs):
        if mode is True and self.config.get("feature_extractor", False) is True and self.feature_extractor is None:
            device, = list(set(p.device for p in self.parameters()))
            self.feature_extractor = VGG16FeatureExtractor()
            self.feature_extractor = self.feature_extractor.to(device=device)
        else:
            self.feature_extractor = None

        super().train(mode=mode, **kwargs)
