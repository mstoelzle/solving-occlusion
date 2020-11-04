from abc import abstractmethod, ABC
import torch
from torch import nn
from typing import Dict, List, Callable, Union, Any, TypeVar, Tuple

from src.learning.models.base_model import BaseModel
from src.enums.channel_enum import ChannelEnum


class BaseVAE(BaseModel, ABC):
    def __init__(self, latent_dim: int, **kwargs) -> None:
        super(BaseVAE, self).__init__(**kwargs)

        self.latent_dim = latent_dim

    def encode(self, input: torch.Tensor) -> Dict:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> Any:
        raise NotImplementedError

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

    def sample(self, batch_size: int, device: torch.device, **kwargs) -> torch.Tensor:
        raise RuntimeWarning()

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
