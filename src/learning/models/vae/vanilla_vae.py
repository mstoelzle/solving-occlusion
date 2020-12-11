import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, List, Callable, Union, Any, TypeVar, Tuple

from .base_vae import BaseVAE
from src.dataloaders.dataloader_meta_info import DataloaderMetaInfo
from src.enums import *
from src.datasets.base_dataset import BaseDataset
from src.learning.loss.loss import *


class VanillaVAE(BaseVAE):
    """
    VanillaVAE
    Implementation strongly inspired by:
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    """

    def __init__(self,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__(**kwargs)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        in_channels = len(self.in_channels)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        sample_data = torch.zeros(size=(1, len(self.in_channels), self.input_dim[0], self.input_dim[1]))
        sample_encoding = self.encoder(sample_data)
        sample_flattened = torch.flatten(sample_encoding, start_dim=1)

        encoding_output_dim = sample_flattened.size(1)

        self.fc_mu = nn.Linear(encoding_output_dim, self.latent_dim)
        self.fc_var = nn.Linear(encoding_output_dim, self.latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, encoding_output_dim)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3, stride=2, padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=len(self.out_channels),
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        resized_height = int(np.sqrt(result.size(1) / self.hidden_dims[-1]))
        result = result.view(result.size(0), self.hidden_dims[-1], resized_height, -1)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, data: Dict[Union[str, ChannelEnum], torch.Tensor],
                **kwargs) -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        input, norm_consts = self.assemble_input(data)

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        rec_dem = self.decode(z).squeeze(dim=1)

        output = {ChannelEnum.REC_DEM: rec_dem,
                  "mu": mu,
                  "log_var": log_var}

        output = self.denormalize_output(data, output, norm_consts)

        return output

    def loss_function(self,
                      loss_config: dict,
                      output: Dict[Union[ChannelEnum, str], torch.Tensor],
                      data: Dict[ChannelEnum, torch.Tensor],
                      dataloader_meta_info: DataloaderMetaInfo = None,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """

        loss_dict = self.eval_loss_function(loss_config=loss_config, output=output, data=data,
                                            dataloader_meta_info=dataloader_meta_info, **kwargs)

        if self.training:
            kld_loss = kld_log_var_loss_fct(output["mu"], output["log_var"])

            weights = loss_config.get("train_weights", {})

            reconstruction_weight = weights.get(LossEnum.MSE_REC_ALL.value, 1)
            reconstruction_occlusion_weight = weights.get(LossEnum.MSE_REC_OCC.value, 1)

            # kld_weight: Account for the minibatch samples from the dataset
            kld_weight = weights.get("kld", None)
            if kld_weight is None:
                kld_weight = data[ChannelEnum.GT_DEM].size(0) / dataloader_meta_info.length

            loss = reconstruction_weight * loss_dict[LossEnum.MSE_REC_ALL] \
                   + reconstruction_occlusion_weight * loss_dict[LossEnum.MSE_REC_OCC] \
                   + kld_weight * kld_loss

            loss_dict.update({LossEnum.LOSS: loss,
                              LossEnum.KLD: kld_loss})

            return loss_dict
        else:
            return loss_dict

    def sample(self,
               num_samples: int,
               device: torch.device, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param device: (torch.device) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(device)

        samples = self.decode(z).squeeze(dim=1)

        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
