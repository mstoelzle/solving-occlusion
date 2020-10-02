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
    def __init__(self, dataset_path: pathlib.Path, purpose: str = None,
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

            if issubclass(type(value), Image.Image):
                value = ToTensor()(value)

                # this code is made for the TrasysPlanetaryDataset
                value = value[0, ...]
            elif issubclass(type(value), np.ndarray):
                # we need to make sure the array is writable
                value = torch.tensor(value)

            if self.transform is not None and key != ChannelEnum.PARAMS:
                value = self.transform(value).squeeze()

            if key == ChannelEnum.BINARY_OCCLUSION_MAP:
                value = value.to(dtype=torch.bool)

            output[key] = value

        if ChannelEnum.BINARY_OCCLUSION_MAP in output and ChannelEnum.OCCLUDED_ELEVATION_MAP not in output:
            output[ChannelEnum.OCCLUDED_ELEVATION_MAP] = self.create_occluded_elevation_map(
                elevation_map=output[ChannelEnum.ELEVATION_MAP],
                binary_occlusion_map=output[ChannelEnum.BINARY_OCCLUSION_MAP])
        elif ChannelEnum.BINARY_OCCLUSION_MAP not in output and ChannelEnum.OCCLUDED_ELEVATION_MAP in output:
            output[ChannelEnum.BINARY_OCCLUSION_MAP] = self.create_binary_occlusion_map(
                occluded_elevation_map=output[ChannelEnum.OCCLUDED_ELEVATION_MAP])
        else:
            raise ValueError

        return output

    def create_occluded_elevation_map(self, elevation_map: torch.Tensor,
                                      binary_occlusion_map: torch.Tensor) -> torch.Tensor:
        occluded_elevation_map = elevation_map.clone()

        occluded_elevation_map[binary_occlusion_map == 1] = np.nan

        return occluded_elevation_map

    def create_binary_occlusion_map(self, occluded_elevation_map: torch.Tensor) -> torch.Tensor:
        binary_occlusion_map = (occluded_elevation_map != occluded_elevation_map)

        return binary_occlusion_map
