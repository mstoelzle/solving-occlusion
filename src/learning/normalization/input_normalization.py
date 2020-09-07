import torch
from typing import *

from src.enums.channel_enum import ChannelEnum


class InputNormalization:
    @staticmethod
    def normalize(channel: ChannelEnum, input: torch.Tensor, batch=True,
                  norm_consts: dict = None) -> Tuple[torch.Tensor, Dict]:
        generate_norm_consts = norm_consts is None

        if channel == ChannelEnum.OCCLUDED_ELEVATION_MAP or channel == ChannelEnum.ELEVATION_MAP:
            if batch:
                normalized_elevation_map = input.clone()
                if generate_norm_consts:
                    norm_consts = {"mean_elevation": input.new_zeros(size=(input.size(0),)),
                                   "stdev_elevation": input.new_zeros(size=(input.size(0),))}

                for idx in range(input.size(0)):
                    item = input[idx, ...]

                    if generate_norm_consts:
                        mean_elevation = torch.mean(item[~torch.isnan(item)])
                        stdev_elevation = torch.std(item[~torch.isnan(item)])
                        norm_consts["mean_elevation"][idx] = mean_elevation
                        norm_consts["stdev_elevation"][idx] = stdev_elevation
                    else:
                        mean_elevation = norm_consts["mean_elevation"][idx]
                        stdev_elevation = norm_consts["stdev_elevation"][idx]

                    normalized_elevation_map[idx, ...][~torch.isnan(item)] = \
                        torch.div(item[~torch.isnan(item)] - mean_elevation, stdev_elevation)
            else:
                raise NotImplementedError
            return normalized_elevation_map, norm_consts
        else:
            return input, norm_consts

    @staticmethod
    def denormalize(channel: ChannelEnum, input: torch.Tensor, norm_consts: Dict, batch=True) -> torch.Tensor:
        if channel == ChannelEnum.OCCLUDED_ELEVATION_MAP or channel == ChannelEnum.ELEVATION_MAP \
                or channel == ChannelEnum.RECONSTRUCTED_ELEVATION_MAP:
            if batch:
                denormalized_elevation_map = input.clone()
                for idx in range(input.size(0)):
                    item = input[idx, ...]
                    mean_elevation = norm_consts["mean_elevation"][idx]
                    stdev_elevation = norm_consts["stdev_elevation"][idx]

                    denormalized_elevation_map[idx, ...][~torch.isnan(item)] = \
                        mean_elevation + torch.mul(item[~torch.isnan(item)], stdev_elevation)
            else:
                raise NotImplementedError
            return denormalized_elevation_map
        else:
            return input
