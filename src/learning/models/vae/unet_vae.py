""" Full assembly of the parts to form the complete network """

import numpy as np
from torch import nn
from typing import *

from .base_vae import BaseVAE
from ..unet.unet_parts import *
from src.dataloaders.dataloader_meta_info import DataloaderMetaInfo
from src.enums import *
from src.datasets.base_dataset import BaseDataset
from src.learning.loss.loss import kld_log_var_loss_fct, total_variation_loss_fct, masked_total_variation_loss_fct


class UNetVAE(BaseVAE):
    def __init__(self, hidden_dims: List = None, bilinear=True, **kwargs):
        super(UNetVAE, self).__init__(**kwargs)

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

        # we send a sample input through the model to infer dynamically the needed size of the fc layers
        sample_input = torch.zeros(size=(1, len(self.in_channels), self.input_dim[0], self.input_dim[1]))
        sample_encodings = self.encode(sample_input)
        sample_x_flat = torch.flatten(sample_encodings[-1], start_dim=1)

        self.fc_mu = nn.Linear(sample_x_flat.size(1), self.latent_dim)
        self.fc_var = nn.Linear(sample_x_flat.size(1), self.latent_dim)
        self.fc_decoder_input = nn.Linear(self.latent_dim, sample_x_flat.size(1))

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

        self.feature_extractor = None

    def forward(self, data: Dict[Union[str, ChannelEnum], torch.Tensor],
                **kwargs) -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        input, norm_consts = self.assemble_input(data)

        encodings = self.encode(input)

        x = encodings[-1]
        x_flat = torch.flatten(x, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        if self.fc_mu is None or self.fc_var is None or self.fc_decoder_input is None:
            device, = list(set(p.device for p in self.parameters()))

            self.fc_mu = nn.Linear(x_flat.size(1), self.latent_dim).to(device=device)
            self.fc_var = nn.Linear(x_flat.size(1), self.latent_dim).to(device=device)
            self.fc_decoder_input = nn.Linear(self.latent_dim, x_flat.size(1)).to(device=device)

        mu = self.fc_mu(x_flat)
        log_var = self.fc_var(x_flat)

        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu

        x = self.fc_decoder_input(z)
        x = x.view(encodings[-1].size(0), encodings[-1].size(1), encodings[-1].size(2), encodings[-1].size(3))
        rec_dem = self.decode(x, encodings).squeeze(dim=1)

        output = {ChannelEnum.REC_DEM: rec_dem, "mu": mu, "log_var": log_var}

        if self.num_solutions > 1 and self.training is False:
            dem_solutions = []
            for i in range(self.num_solutions):
                z = self.reparameterize(mu, log_var)

                x = self.fc_decoder_input(z)
                x = x.view(encodings[-1].size(0), encodings[-1].size(1), encodings[-1].size(2), encodings[-1].size(3))
                x = self.decode(x, encodings).squeeze(dim=1)

                dem_solutions.append(x)

            dem_solutions = torch.stack(dem_solutions, dim=1)
            model_uncertainty = torch.var(dem_solutions, dim=1)

            output[ChannelEnum.REC_DEMS] = dem_solutions
            output[ChannelEnum.MODEL_UM] = model_uncertainty
            output[ChannelEnum.TOTAL_UM] = model_uncertainty

            if self.use_mean_as_rec:
                output[ChannelEnum.REC_DEM] = torch.mean(dem_solutions, dim=1)

        output = self.denormalize_output(data, output, norm_consts)

        return output

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        encodings = []
        for encoding_idx, encoder_layer in enumerate(self.encoder):
            if len(encodings) == 0:
                encodings.append(encoder_layer(input))
            else:
                encodings.append(encoder_layer(encodings[-1]))

        return encodings

    def decode(self, input: torch.Tensor, encodings: List[torch.Tensor]) -> torch.Tensor:
        reversed_encodings = encodings.copy()
        reversed_encodings.reverse()

        x = input
        for decoding_idx, decoder_layer in enumerate(self.decoder):
            if decoding_idx + 1 < len(self.decoder):
                x = decoder_layer(x, reversed_encodings[decoding_idx+1])
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
                                            dataloader_meta_info=dataloader_meta_info,**kwargs)

        if self.training:
            weights = loss_config.get("train_weights", {})

            reconstruction_weight = weights.get(LossEnum.MSE_REC_ALL.value, 0)
            reconstruction_non_occlusion_weight = weights.get(LossEnum.MSE_REC_NOCC.value, 1)
            reconstruction_occlusion_weight = weights.get(LossEnum.MSE_REC_OCC.value, 1)
            perceptual_weight = weights.get(LossEnum.PERCEPTUAL.value, 0)
            style_weight = weights.get(LossEnum.STYLE.value, 0)
            total_variation_weight = weights.get(LossEnum.TV.value, 0)

            # kld_weight: Account for the minibatch samples from the dataset
            kld_weight = weights.get("kld", None)
            if kld_weight is None:
                kld_weight = data[ChannelEnum.GT_DEM].size(0) / dataloader_meta_info.length

            if perceptual_weight > 0 or style_weight > 0:
                artistic_loss = self.artistic_loss_function(loss_config=loss_config, output=output, data=data, **kwargs)
                loss_dict.update(artistic_loss)
            total_variation_loss = masked_total_variation_loss_fct(input=output[ChannelEnum.COMP_DEM],
                                                                   mask=data[ChannelEnum.OCC_MASK])

            kld_loss = kld_log_var_loss_fct(output["mu"], output["log_var"])

            loss = reconstruction_weight * loss_dict[LossEnum.MSE_REC_ALL] \
                   + reconstruction_non_occlusion_weight * loss_dict[LossEnum.MSE_REC_NOCC] \
                   + reconstruction_occlusion_weight * loss_dict[LossEnum.MSE_REC_OCC] \
                   + perceptual_weight * loss_dict.get(LossEnum.PERCEPTUAL, 0.) \
                   + style_weight * loss_dict.get(LossEnum.STYLE, 0.) \
                   + total_variation_weight * total_variation_loss \
                   + kld_weight * kld_loss

            loss_dict.update({LossEnum.LOSS: loss})

            return loss_dict
        else:
            return loss_dict

    def train(self,  mode: bool = True):
        if mode is True and self.config.get("feature_extractor", False) is True:
            device, = list(set(p.device for p in self.parameters()))
            self.feature_extractor = VGG16FeatureExtractor()
            self.feature_extractor = self.feature_extractor.to(device=device)
        else:
            self.feature_extractor = None

        super().train(mode=mode)
