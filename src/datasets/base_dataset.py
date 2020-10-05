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
    def __init__(self, config: dict, dataset_path: pathlib.Path, purpose: str = None,
                 transform: Optional[Callable] = None, transforms: Optional[Callable] = None):
        super().__init__(root=str(dataset_path), transform=transform, transforms=transforms)

        self.config = config
        self.purpose = purpose
        self.dataset_path = dataset_path

        self.img_loader = torchvision_default_loader

    def prepare_item(self, data: dict, trasys: bool = False) -> Dict[ChannelEnum, torch.Tensor]:
        for key, value in data.items():
            if type(key) == str:
                new_key = ChannelEnum(key)
                data[new_key] = value
                del data[key]

        output = {}
        for key, value in data.items():
            if issubclass(type(value), pathlib.Path):
                value = self.img_loader(value)

            if issubclass(type(value), Image.Image):
                value = ToTensor()(value)

                # this code is made for the TrasysPlanetaryDataset
                value = value[0, ...]
            elif issubclass(type(value), np.ndarray):
                value = torch.tensor(value)

            if key == ChannelEnum.BINARY_OCCLUSION_MAP:
                value = value.to(dtype=torch.bool)

            output[key] = value

        if ChannelEnum.BINARY_OCCLUSION_MAP not in output and ChannelEnum.OCCLUDED_ELEVATION_MAP in output:
            output[ChannelEnum.BINARY_OCCLUSION_MAP] = self.create_binary_occlusion_map(
                occluded_elevation_map=output[ChannelEnum.OCCLUDED_ELEVATION_MAP])

        # we require square dimension for now
        assert output[ChannelEnum.ELEVATION_MAP].size(0) == output[ChannelEnum.ELEVATION_MAP].size(1)
        if type(self.config["size"]) == list:
            assert self.config["size"][0] == self.config["size"][1]
        input_size = output[ChannelEnum.ELEVATION_MAP].size(0)

        if trasys is True:
            # TODO: add actual params from dataset metadata
            terrain_resolution = 200. / 128  # 200m terrain length divided by 128 pixels
            camera_elevation = 2.  # Camera is elevated on 2m

            output[ChannelEnum.PARAMS] = torch.tensor([terrain_resolution, 0., 0., camera_elevation, 0.])

            # the binary occlusion mask is inverse for the trasys planetary dataset
            output[ChannelEnum.BINARY_OCCLUSION_MAP] = ~output[ChannelEnum.BINARY_OCCLUSION_MAP]

            # the encoded elevation map of the trasys dataset measures the orthogonal distance
            # from the camera to the terrain
            output[ChannelEnum.ELEVATION_MAP] = camera_elevation - output[ChannelEnum.ELEVATION_MAP] * 255

        # apply transforms
        if self.transform is not None:
            for key, value in output.items():
                if key == ChannelEnum.PARAMS:
                    # we need to apply the resizing to the terrain resolution
                    input_resolution = value[0].item()
                    if type(self.config["size"]) == list:
                        output_size = self.config["size"][0]
                    elif type(self.config["size"]) == int:
                        output_size = self.config["size"]
                    else:
                        raise ValueError

                    output_resolution = input_resolution * input_size / output_size
                    value[0] = output_resolution
                elif key != ChannelEnum.OCCLUDED_ELEVATION_MAP:
                    if value.dtype == torch.bool:
                        value = value.to(dtype=torch.int)

                    value = self.transform(value).squeeze()
                else:
                    continue

                output[key] = value

        if ChannelEnum.BINARY_OCCLUSION_MAP in output:
            output[ChannelEnum.BINARY_OCCLUSION_MAP] = output[ChannelEnum.BINARY_OCCLUSION_MAP].to(dtype=torch.bool)
            output[ChannelEnum.OCCLUDED_ELEVATION_MAP] = self.create_occluded_elevation_map(
                elevation_map=output[ChannelEnum.ELEVATION_MAP],
                binary_occlusion_map=output[ChannelEnum.BINARY_OCCLUSION_MAP])
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
