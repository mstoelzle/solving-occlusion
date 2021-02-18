import torch
from typing import *

from src.enums.channel_enum import ChannelEnum


class InputNormalization:
    @staticmethod
    def normalize(channel: ChannelEnum, input: torch.Tensor, mean: bool = True, stdev: bool = True,
                  batch=True, norm_consts: dict = None, **kwargs) -> Tuple[torch.Tensor, Dict]:
        generate_norm_consts = norm_consts is None

        if channel == ChannelEnum.OCC_DEM or channel == ChannelEnum.GT_DEM \
                or channel == ChannelEnum.REC_DEM:
            if batch:
                normalized_elevation_map = input.clone()
                if generate_norm_consts:
                    norm_consts = {"mean_elevation": input.new_zeros(size=(input.size(0),)),
                                   "stdev_elevation": input.new_zeros(size=(input.size(0),))}

                for idx in range(input.size(0)):
                    item = input[idx, ...]

                    if generate_norm_consts:
                        if torch.isnan(item).all():
                            mean_elevation = 0.
                            stdev_elevation = 1.
                        else:
                            mean_elevation = torch.mean(item[~torch.isnan(item)])
                            stdev_elevation = torch.std(item[~torch.isnan(item)])

                        norm_consts["mean_elevation"][idx] = mean_elevation
                        norm_consts["stdev_elevation"][idx] = stdev_elevation
                    else:
                        mean_elevation = norm_consts["mean_elevation"][idx]
                        stdev_elevation = norm_consts["stdev_elevation"][idx]

                    if mean is True and stdev is True:
                        normalized_elevation_map[idx, ...][~torch.isnan(item)] = \
                            torch.div(item[~torch.isnan(item)] - mean_elevation, stdev_elevation)
                    elif mean is True and stdev is False:
                        normalized_elevation_map[idx, ...][~torch.isnan(item)] = \
                            item[~torch.isnan(item)] - mean_elevation
                    elif mean is False and stdev is False:
                        # we just return the unmodified elevation map
                        pass
                    else:
                        raise ValueError
            else:
                raise NotImplementedError
            return normalized_elevation_map, norm_consts
        else:
            return input, norm_consts

    @staticmethod
    def denormalize(channel: ChannelEnum, input: torch.Tensor, norm_consts: Dict,
                    mean: bool = True, stdev: bool = True, batch=True, **kwargs) -> torch.Tensor:
        if channel == ChannelEnum.OCC_DEM or channel == ChannelEnum.GT_DEM \
                or channel == ChannelEnum.REC_DEM:
            if batch:
                denormalized_elevation_map = input.clone()
                for idx in range(input.size(0)):
                    item = input[idx, ...]
                    mean_elevation = norm_consts["mean_elevation"][idx]
                    stdev_elevation = norm_consts["stdev_elevation"][idx]

                    if mean is True and stdev is True:
                        denormalized_elevation_map[idx, ...][~torch.isnan(item)] = \
                            mean_elevation + torch.mul(item[~torch.isnan(item)], stdev_elevation)
                    elif mean is True and stdev is False:
                        denormalized_elevation_map[idx, ...][~torch.isnan(item)] = \
                            mean_elevation + item[~torch.isnan(item)]
                    elif mean is False and stdev is False:
                        # we just return the unmodified elevation map
                        pass
                    else:
                        raise ValueError
            else:
                raise NotImplementedError
            return denormalized_elevation_map
        else:
            return input
