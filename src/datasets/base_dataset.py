from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple

from src.enums.channel_enum import ChannelEnum
from src.utils.log import get_logger

logger = get_logger("base_dataset")


class BaseDataset(ABC):
    def __init__(self, purpose: str):
        self.purpose = purpose

    def prepare_item(self, **kwargs) -> Dict[ChannelEnum, torch.Tensor]:
        output = {}
        for key, value in kwargs.items():
            if type(key) == str:
                key = ChannelEnum(key)

            value = torch.tensor(value)

            output[key] = value

        return output
