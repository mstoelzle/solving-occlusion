from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import Dict, List, Callable, Union, Any, TypeVar, Tuple


class BaseModel(ABC, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass
