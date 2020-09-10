import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, List, Callable, Union, Any, TypeVar, Tuple

from src.learning.models import BaseVAE
from src.enums import *
from src.learning.loss.loss import kld_loss_fct, reconstruction_occlusion_loss_fct
from src.learning.normalization.input_normalization import InputNormalization


class VanillaVAE(BaseVAE):

    def __init__(self,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__(**kwargs)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

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
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
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
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, data: Dict[Union[str, ChannelEnum], torch.Tensor],
                **kwargs) -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        input, norm_consts = self.assemble_input(data)

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        reconstructed_elevation_map = self.decode(z).squeeze()

        if self.input_normalization:
            reconstructed_elevation_map = InputNormalization.denormalize(ChannelEnum.RECONSTRUCTED_ELEVATION_MAP,
                                                                         input=reconstructed_elevation_map,
                                                                         batch=True,
                                                                         norm_consts=norm_consts[
                                                                             ChannelEnum.OCCLUDED_ELEVATION_MAP])

        output = {ChannelEnum.RECONSTRUCTED_ELEVATION_MAP: reconstructed_elevation_map,
                  "mu": mu,
                  "log_var": log_var}

        return output

    def loss_function(self,
                      loss_config: dict,
                      output: Dict[Union[ChannelEnum, str], torch.Tensor],
                      data: Dict[ChannelEnum, torch.Tensor],
                      dataset_length: int,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """

        kld_weight = self.config.get("kld_weight", None)
        if kld_weight is None:
            kld_weight = data[ChannelEnum.ELEVATION_MAP].size(0) / dataset_length

        elevation_map = data[ChannelEnum.ELEVATION_MAP]
        reconstructed_elevation_map = output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP]
        binary_occlusion_map = self.create_binary_occlusion_map(data[ChannelEnum.OCCLUDED_ELEVATION_MAP])

        if LossEnum.RECONSTRUCTION.value in loss_config.get("normalization", []):
            elevation_map, ground_truth_norm_consts = InputNormalization.normalize(ChannelEnum.ELEVATION_MAP,
                                                                                   input=elevation_map,
                                                                                   batch=True)
            reconstructed_elevation_map, _ = InputNormalization.normalize(ChannelEnum.RECONSTRUCTED_ELEVATION_MAP,
                                                                          input=reconstructed_elevation_map,
                                                                          batch=True,
                                                                          norm_consts=ground_truth_norm_consts)

        reconstruction_loss = F.mse_loss(reconstructed_elevation_map, elevation_map)
        reconstruction_occlusion_loss = reconstruction_occlusion_loss_fct(reconstructed_elevation_map,
                                                                          elevation_map,
                                                                          binary_occlusion_map)

        kld_loss = kld_loss_fct(output["mu"], output["log_var"])

        if self.training:
            # kld_weight: Account for the minibatch samples from the dataset
            loss = reconstruction_loss + reconstruction_occlusion_loss + kld_weight * kld_loss

            return {LossEnum.LOSS: loss,
                    LossEnum.RECONSTRUCTION: reconstruction_loss,
                    LossEnum.RECONSTRUCTION_OCCLUSION: reconstruction_occlusion_loss,
                    LossEnum.KLD: -kld_loss}
        else:
            return self.eval_loss_function(loss_config=loss_config, output=output, data=data, **kwargs)

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

        samples = self.decode(z).squeeze()

        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
