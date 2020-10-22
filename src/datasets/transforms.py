import numpy as np
import torch

from src.enums import *


class Transformer:
    def __init__(self, purpose: str, transforms: list):
        self.purpose = purpose

        self.transforms = transforms
        self.rng = np.random.RandomState(seed=1)

    def __call__(self, input: torch.Tensor, channel: ChannelEnum = None) -> torch.Tensor:
        transformed_input = input

        for transform_config in self.transforms:
            if transform_config["type"] == "random_noise":
                transformed_input = self.random_noise(transform_config, transformed_input, channel=channel)
            elif transform_config["type"] == "random_scale":
                transformed_input = self.random_scale(transform_config, transformed_input, channel=channel)
            elif transform_config["type"] == "random_offset":
                transformed_input = self.random_offset(transform_config, transformed_input, channel=channel)
            else:
                raise NotImplementedError

        return transformed_input

    def random_noise(self, transform_config: dict, input: np.array, channel: ChannelEnum = None) -> np.array:
        if channel in [None, ChannelEnum.OCCLUDED_ELEVATION_MAP]:
            stdev = transform_config["stdev"]

            deterministic = transform_config.get("deterministic", True)
            if deterministic:
                noise = self.rng.normal(loc=0, scale=stdev, size=tuple(input.size()))
            else:
                noise = np.random.normal(loc=0, scale=stdev, size=tuple(input.size()))

            noise = input.new_tensor(noise, dtype=input.dtype)

            transformed_input = input + noise
            return transformed_input

        return input

    def random_scale(self, transform_config: dict, input: np.array, channel: ChannelEnum = None) -> np.array:
        if channel in [None, ChannelEnum.OCCLUDED_ELEVATION_MAP, ChannelEnum.GROUND_TRUTH_ELEVATION_MAP]:
            min, max = transform_config["min"], transform_config["max"]

            deterministic = transform_config.get("deterministic", True)
            if deterministic:
                scale = self.rng.uniform(low=min, high=max)
            else:
                scale = np.random.uniform(low=min, high=max)

            transformed_input = scale * input
            return transformed_input

        return input

    def random_offset(self, transform_config: dict, input: np.array, channel: ChannelEnum = None) -> np.array:
        if channel in [None, ChannelEnum.OCCLUDED_ELEVATION_MAP, ChannelEnum.GROUND_TRUTH_ELEVATION_MAP]:
            min, max = transform_config["min"], transform_config["max"]

            deterministic = transform_config.get("deterministic", True)
            if deterministic:
                offset = self.rng.uniform(low=min, high=max)
            else:
                offset = np.random.uniform(low=min, high=max)

            transformed_input = offset + input
            return transformed_input

        return input