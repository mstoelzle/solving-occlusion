from abc import ABC, abstractmethod
import numpy as np
import pathlib
from PIL import Image
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset
import torchvision
from typing import *
import warnings

from src.enums.channel_enum import ChannelEnum
from src.utils.log import get_logger

logger = get_logger("base_dataset")


class BaseDataset(ABC):
    def __init__(self, config: dict, dataset_path: pathlib.Path, purpose: str = None,
                 transform: Optional[Callable] = None):
        self.config = config
        self.purpose = purpose
        self.dataset_path = dataset_path

        self.transform = transform

        self.min: float = None
        self.max: float = None

        logger.info(f"Initialise dataset with purpose {purpose}")

    @staticmethod
    def prepare_keys(data: dict) -> np.array:
        for key, value in data.items():
            if type(key) == str:
                new_key = ChannelEnum(key)
                data[new_key] = value
                del data[key]
        return data

    def prepare_item(self, data: dict, invert_mask: bool = False) -> Dict[ChannelEnum, torch.Tensor]:
        output = {}
        for key, value in data.items():
            if issubclass(type(value), pathlib.Path):
                if value.suffix in [".tif", ".tiff", ".TIF", ".TIFF"]:
                    value = tifffile.imread(str(value))
                else:
                    value = torchvision.io.read_image(str(value))

            if issubclass(type(value), Image.Image):
                value = torch.tensor(np.array(value))
                if value.dim() > 2:
                    value = value[0, ...]
            elif issubclass(type(value), np.ndarray):
                value = torch.tensor(value)

            if key == ChannelEnum.OCC_MASK:
                if value.dim() == 3:
                    value = value[0, ...]  # we are only using the first channel if we get a channel-wise input

                value = value.to(dtype=torch.bool)

            output[key] = value

        if ChannelEnum.OCC_MASK not in output and ChannelEnum.OCC_DEM in output:
            output[ChannelEnum.OCC_MASK] = self.create_binary_occlusion_map(
                occluded_elevation_map=output[ChannelEnum.OCC_DEM])
        else:
            if invert_mask:
                output[ChannelEnum.OCC_MASK] = ~output[ChannelEnum.OCC_MASK]

        # we require square dimension for now
        if ChannelEnum.GT_DEM in output:
            sample_map = output[ChannelEnum.GT_DEM]
        elif ChannelEnum.OCC_MASK in output:
            sample_map = output[ChannelEnum.OCC_MASK]
        else:
            raise ValueError

        assert sample_map.size(0) == sample_map.size(1)
        if type(self.config["size"]) == list:
            assert self.config["size"][0] == self.config["size"][1]
        input_size = sample_map.size(0)
        if type(self.config["size"]) == list:
            output_size = self.config["size"][0]
        elif type(self.config["size"]) == int:
            output_size = self.config["size"]
        else:
            raise ValueError

        if input_size != output_size:
            for key, value in output.items():
                if key == ChannelEnum.PARAMS:
                    # we need to apply the resizing to the terrain resolution
                    input_resolution = value[0].item()

                    output_resolution = input_resolution * input_size / output_size
                    value[0] = output_resolution
                else:
                    channel_is_bool = False
                    if value.dtype == torch.bool:
                        channel_is_bool = True
                        value = value.to(dtype=torch.float)

                    interpolation_input = value.unsqueeze(dim=0).unsqueeze(dim=0)
                    interpolation_output = F.interpolate(interpolation_input, size=self.config["size"])
                    value = interpolation_output.squeeze()

                    if channel_is_bool is True:
                        value = value.to(dtype=torch.bool)

                output[key] = value

        if ChannelEnum.OCC_MASK in output and ChannelEnum.GT_DEM in output:
            output[ChannelEnum.OCC_DEM] = \
                self.create_occluded_elevation_map(elevation_map=output[ChannelEnum.GT_DEM],
                                                   occ_mask=output[ChannelEnum.OCC_MASK])

        elif ChannelEnum.OCC_DEM in output:
            pass
        else:
            raise ValueError

        if self.config.get("self_supervision", False) is True:
            if ChannelEnum.GT_DEM in output:
                warnings.warn("Overwriting the ground-truth in the dataset as self-supervision is activated.")

            output[ChannelEnum.GT_DEM] = output[ChannelEnum.OCC_DEM].clone()

        if self.transform is not None:
            output = self.transform(output)

        return output

    def create_occluded_elevation_map(self, elevation_map: torch.Tensor,
                                      occ_mask: torch.Tensor) -> torch.Tensor:
        occluded_elevation_map = elevation_map.clone()

        occluded_elevation_map[occ_mask == 1] = np.nan

        return occluded_elevation_map

    def create_binary_occlusion_map(self, occluded_elevation_map: torch.Tensor) -> torch.Tensor:
        occ_mask = (occluded_elevation_map != occluded_elevation_map)

        return occ_mask
