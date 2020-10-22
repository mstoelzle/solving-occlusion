import numpy as np
import torch

from src.enums import *


class Transformer:
    def __init__(self, purpose: str, transforms: list):
        self.purpose = purpose

        self.transforms = transforms
        self.rng = np.random.RandomState(seed=1)

        self.deterministic = False
        if self.purpose == "test":
            self.deterministic = True

    def __call__(self, input: torch.Tensor, channel: ChannelEnum = None) -> torch.Tensor:
        transformed_input = input

        for transform_config in self.transforms:
            if transform_config["type"] == "random_noise":
                transformed_input = self.random_noise(transform_config, transformed_input, channel=channel)
            elif transform_config["type"] == "random_vertical_scale":
                transformed_input = self.random_vertical_scale(transform_config, transformed_input, channel=channel)
            elif transform_config["type"] == "random_vertical_offset":
                transformed_input = self.random_vertical_offset(transform_config, transformed_input, channel=channel)
            else:
                raise NotImplementedError

        return transformed_input

    def random_noise(self, transform_config: dict, input: np.array, channel: ChannelEnum = None) -> np.array:
        if channel is None or channel.value in transform_config["apply_to"]:
            stdev = transform_config["stdev"]

            if self.deterministic:
                noise = self.rng.normal(loc=0, scale=stdev, size=tuple(input.size()))
            else:
                noise = np.random.normal(loc=0, scale=stdev, size=tuple(input.size()))

            noise = input.new_tensor(noise, dtype=input.dtype)

            transformed_input = input + noise
            return transformed_input

        return input

    def random_vertical_scale(self, transform_config: dict, input: np.array, channel: ChannelEnum = None) -> np.array:
        if channel is None or channel.value in transform_config["apply_to"]:
            min, max = transform_config["min"], transform_config["max"]

            if self.deterministic:
                scale = self.rng.uniform(low=min, high=max)
            else:
                scale = np.random.uniform(low=min, high=max)

            if channel is ChannelEnum.PARAMS:
                transformed_input = input.clone()
                transformed_input[3] = scale * transformed_input[3]
            else:
                transformed_input = scale * input

            return transformed_input

        return input

    def random_vertical_offset(self, transform_config: dict, input: np.array, channel: ChannelEnum = None) -> np.array:
        if channel is None or channel.value in transform_config["apply_to"]:
            min, max = transform_config["min"], transform_config["max"]

            if self.deterministic:
                offset = self.rng.uniform(low=min, high=max)
            else:
                offset = np.random.uniform(low=min, high=max)

            if channel is ChannelEnum.PARAMS:
                transformed_input = input.clone()
                transformed_input[3] = offset + transformed_input[3]
            else:
                transformed_input = offset + input

            return transformed_input

        return input
