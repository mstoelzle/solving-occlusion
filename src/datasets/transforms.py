import numpy as np
import torch
from typing import *

from src.enums import *


class Transformer:
    def __init__(self, purpose: str, transforms: list):
        self.purpose = purpose

        self.transforms = transforms
        self.rng = np.random.RandomState(seed=1)

        self.deterministic = False
        if self.purpose == "test":
            self.deterministic = True

    def __call__(self, data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        transformed_data = data

        for transform_config in self.transforms:
            if transform_config["type"] == "random_noise":
                transformed_data = self.random_noise(transform_config, transformed_data)
            elif transform_config["type"] == "random_vertical_scale":
                transformed_data = self.random_vertical_scale(transform_config, transformed_data)
            elif transform_config["type"] == "random_vertical_offset":
                transformed_data = self.random_vertical_offset(transform_config, transformed_data)
            elif transform_config["type"] == "random_occlusion":
                transformed_data = self.random_occlusion(transform_config, transformed_data)
            else:
                raise NotImplementedError

        return transformed_data

    def random_noise(self, transform_config: dict,
                     data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        stdev = transform_config["stdev"]

        noise = None
        for channel, value in data.items():
            if channel.value in transform_config["apply_to"]:
                if noise is None:
                    if self.deterministic:
                        noise_value = self.rng.normal(loc=0, scale=stdev, size=tuple(value.size()))
                    else:
                        noise_value = np.random.normal(loc=0, scale=stdev, size=tuple(value.size()))
                    noise = value.new_tensor(noise_value, dtype=value.dtype)

                transformed_value = value + noise

                data[channel] = transformed_value

        return data

    def random_vertical_scale(self, transform_config: dict,
                              data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        min, max = transform_config["min"], transform_config["max"]

        if self.deterministic:
            scale = self.rng.uniform(low=min, high=max)
        else:
            scale = np.random.uniform(low=min, high=max)

        for channel, value in data.items():
            if channel.value in transform_config["apply_to"]:
                if channel is ChannelEnum.PARAMS:
                    transformed_value = value.clone()
                    transformed_value[3] = scale * transformed_value[3]
                else:
                    transformed_value = scale * value

                data[channel] = transformed_value

        return data

    def random_vertical_offset(self, transform_config: dict,
                               data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        min, max = transform_config["min"], transform_config["max"]

        if self.deterministic:
            offset = self.rng.uniform(low=min, high=max)
        else:
            offset = np.random.uniform(low=min, high=max)

        for channel, value in data.items():
            if channel.value in transform_config["apply_to"]:
                if channel is ChannelEnum.PARAMS:
                    transformed_value = value.clone()
                    transformed_value[3] = offset + transformed_value[3]
                else:
                    transformed_value = offset + value

                data[channel] = transformed_value

        return data

    def random_occlusion(self, transform_config: dict,
                         data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        probability = transform_config["probability"]

        occlusion = None
        for channel, value in data.items():
            if channel.value in transform_config["apply_to"]:
                if occlusion is None:
                    if self.deterministic:
                        occlusion = self.rng.choice([0, 1], p=[1-probability, probability], size=tuple(value.size()))
                    else:
                        occlusion = np.random.choice([0, 1], p=[1-probability, probability], size=tuple(value.size()))
                    occlusion = torch.tensor(occlusion)

                if channel == ChannelEnum.BINARY_OCCLUSION_MAP:
                    # assert value.dtype == torch.bool
                    #
                    # transformed_value = value.to(dtype=torch.int) + occlusion
                    #
                    # transformed_value[transformed_value > 1] = 1
                    #
                    # data[channel] = transformed_value.to(dtype=torch.bool)

                    value[occlusion == 1] = True
                elif channel == ChannelEnum.OCCLUDED_ELEVATION_MAP:
                    value[occlusion == 1] = np.nan
                else:
                    raise NotImplementedError

        return data
