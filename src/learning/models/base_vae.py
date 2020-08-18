from abc import abstractmethod, ABC
import torch
from torch import nn
from typing import Dict, List, Callable, Union, Any, TypeVar, Tuple

from .base_model import BaseModel


class BaseVAE(BaseModel, ABC):
    def __init__(self, **kwargs) -> None:
        super(BaseVAE, self).__init__(**kwargs)

    def encode(self, input: torch.Tensor) -> Dict:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, device: torch.device, **kwargs) -> torch.Tensor:
        raise RuntimeWarning()

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
