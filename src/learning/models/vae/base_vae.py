from abc import abstractmethod, ABC
import torch
from torch import nn
from typing import Dict, List, Callable, Union, Any, TypeVar, Tuple

from src.learning.models.base_model import BaseModel
from src.enums.channel_enum import ChannelEnum


class BaseVAE(BaseModel, ABC):
    def __init__(self, in_channels: List[str], out_channels: List[str], latent_dim: int, **kwargs) -> None:
        super(BaseVAE, self).__init__(**kwargs)

        self.in_channels = [ChannelEnum(in_channel) for in_channel in in_channels]
        self.out_channels = [ChannelEnum(out_channel) for out_channel in out_channels]
        self.latent_dim = latent_dim

    def encode(self, input: torch.Tensor) -> Dict:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, device: torch.device, **kwargs) -> torch.Tensor:
        raise RuntimeWarning()

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
