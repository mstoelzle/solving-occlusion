from abc import ABC, abstractmethod
import numpy as np
import pathlib
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset
from torchvision.transforms import ToTensor, Resize
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader as torchvision_default_loader
from typing import *

from src.enums.channel_enum import ChannelEnum
from src.utils.log import get_logger

logger = get_logger("base_dataset")


class BaseDataset(VisionDataset):
    def __init__(self, purpose: str, dataset_path: pathlib.Path,
                 transform: Optional[Callable] = None, transforms: Optional[Callable] = None):
        super().__init__(root=str(dataset_path), transform=transform, transforms=transforms)

        self.purpose = purpose
        self.dataset_path = dataset_path

        self.img_loader = torchvision_default_loader

    def prepare_item(self, data: dict) -> Dict[ChannelEnum, torch.Tensor]:
        output = {}
        for key, value in data.items():
            if type(key) == str:
                key = ChannelEnum(key)

            if issubclass(type(value), pathlib.Path):
                value = self.img_loader(value)

            if self.transform is not None and issubclass(type(value), Image.Image):
                value = self.transform(value)

            if type(value) != torch.Tensor:
                if issubclass(type(value), Image.Image):
                    w, h = value.size
                    if w != 64 or h != 64:
                        value = Resize(size=(64, 64))(value)

                    value = ToTensor()(value)

                    # this code is made for the TrasysPlanetaryDataset
                    value = value[0, ...]
                else:
                    value = torch.tensor(value)

            output[key] = value

        return output
