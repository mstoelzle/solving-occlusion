from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import *

from src.enums.channel_enum import ChannelEnum
from src.learning.normalization.input_normalization import InputNormalization


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

    def assemble_input(self, data: Dict[Union[str, ChannelEnum], torch.Tensor]) -> torch.Tensor:
        input = None
        for channel_idx, in_channel in enumerate(self.in_channels):
            if in_channel in data:
                channel_data = data[in_channel]
            elif in_channel == ChannelEnum.BINARY_OCCLUSION_MAP:
                channel_data = self.create_binary_occlusion_map(data[ChannelEnum.OCCLUDED_ELEVATION_MAP])
            else:
                raise NotImplementedError

            if in_channel == ChannelEnum.OCCLUDED_ELEVATION_MAP or in_channel == ChannelEnum.ELEVATION_MAP:
                channel_data, norm_consts = InputNormalization.normalize(in_channel, channel_data, batch=True)

            if in_channel == ChannelEnum.OCCLUDED_ELEVATION_MAP:
                channel_data = self.preprocess_occluded_elevation_map(channel_data)

            if input is None:
                input = channel_data.new_zeros(size=(channel_data.size(0), len(self.in_channels),
                                                     channel_data.size(1), channel_data.size(2)))

            input[:, channel_idx, ...] = channel_data.unsqueeze(dim=1)[:, 0, :, :]

        return input

    def preprocess_occluded_elevation_map(self, occluded_elevation_map: torch.Tensor) -> torch.Tensor:
        poem = occluded_elevation_map.clone()

        # replace NaNs signifying occluded areas with arbitrary high or low number
        poem[occluded_elevation_map != occluded_elevation_map] = -10000

        return poem

    def create_binary_occlusion_map(self, occluded_elevation_map: torch.Tensor) -> torch.Tensor:
        binary_occlusion_map = (occluded_elevation_map != occluded_elevation_map)

        return binary_occlusion_map
