import numpy as np
import os
from pathlib import Path
import scipy
import torch
from torch.nn import functional as F
from typing import *

from src.enums import *


class Transformer:
    def __init__(self, purpose: str, transforms: list):
        self.purpose = purpose

        self.transforms = transforms

        if self.purpose == "test":
            self.deterministic = True
        else:
            self.deterministic = False

        self.rng = np.random.RandomState(seed=1)

        try:
            import grid_map_raycasting
            self.grid_map_raycasting = grid_map_raycasting

            # We need to set the path to the raisim license file
            # It is usually placed in the rsc folder of the raisim installation
            # Path to raisim install (e.g. $LOCAL_INSTALL) as saved in the environment variable $RAISIM_INSTALL
            # run this in your terminal: export RAISIM_INSTALL=$LOCAL_INSTALL
            grid_map_raycasting.setRaisimLicenseFile(str(Path(os.getenv("RAISIM_INSTALL"))/"rsc"/"activation.raisim"))
        except ImportError:
            self.grid_map_raycasting = None


    def __call__(self, data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        transformed_data = data

        for transform_config in self.transforms:
            transform_type = TransformEnum(transform_config["type"])

            transform_fct = getattr(self, transform_type.value)
            transformed_data = transform_fct(transform_config, transformed_data)

        return transformed_data

    def white_noise(self, transform_config: dict,
                    data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        stdev = transform_config["stdev"]

        noise = None
        for channel, value in data.items():
            if channel.value in transform_config["apply_to"]:
                if noise is None:
                    noise_value = self.rng.normal(loc=0, scale=stdev, size=tuple(value.size()))
                    noise = value.new_tensor(noise_value, dtype=value.dtype)

                transformed_value = value + noise

                data[channel] = transformed_value

        return data

    def range_adjusted_white_noise(self, transform_config: dict,
                                   data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        rng = self.rng if self.deterministic else np.random

        res_grid = data[ChannelEnum.RES_GRID]
        robot_position = data[ChannelEnum.REL_POSITION]

        sample_grid = data[ChannelEnum.OCC_DEM]

        # distance of every pixel from the robot
        lin_x = np.arange(start=-sample_grid.shape[0] / 2, stop=sample_grid.shape[0] / 2, step=1) * res_grid[0].numpy()
        lin_y = np.arange(start=-sample_grid.shape[1] / 2, stop=sample_grid.shape[1] / 2, step=1) * res_grid[1].numpy()
        off_y, off_x = np.meshgrid(lin_x, lin_y)

        dist_x = off_x - robot_position[0].numpy()
        dist_y = off_y - robot_position[1].numpy()
        dist_p2_norm = np.sqrt(np.square(dist_x) + np.square(dist_y))

        stdev = transform_config["stdev"]
        range = transform_config["range"]

        scale = stdev * np.square(1 / range * dist_p2_norm)

        noise = None
        for channel, value in data.items():
            if channel.value in transform_config["apply_to"]:
                if noise is None:
                    noise_value = rng.normal(loc=0, scale=scale, size=tuple(value.size()))
                    noise = value.new_tensor(noise_value, dtype=value.dtype)

                transformed_value = value + noise

                data[channel] = transformed_value

        return data

    def gaussian_filtered_white_noise(self, transform_config: dict,
                                      data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        rng = self.rng if self.deterministic else np.random

        probability = transform_config["probability"]
        horizontal_stdev = transform_config["horizontal_stdev"]
        vertical_stdev = transform_config["vertical_stdev"]

        filtered_noise = None
        for channel, value in data.items():
            if channel.value in transform_config["apply_to"]:
                if filtered_noise is None:
                    peaks = rng.choice([0, 1], p=[1 - probability, probability], size=tuple(value.size()))
                    white_noise = rng.normal(loc=0, scale=vertical_stdev, size=tuple(value.size()))
                    noise = peaks * white_noise
                    filtered_noise = scipy.ndimage.gaussian_filter(input=noise, sigma=horizontal_stdev)

                    filtered_noise = value.new_tensor(filtered_noise, dtype=value.dtype)

                transformed_value = value + filtered_noise

                data[channel] = transformed_value

        return data

    def random_vertical_scale(self, transform_config: dict,
                              data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        rng = self.rng if self.deterministic else np.random

        min, max = transform_config["min"], transform_config["max"]

        scale = rng.uniform(low=min, high=max)

        for channel, value in data.items():
            if channel.value in transform_config["apply_to"]:
                if channel is ChannelEnum.REL_POSITION:
                    transformed_value = value.clone()
                    transformed_value[2] = scale * transformed_value[2]
                else:
                    transformed_value = scale * value

                data[channel] = transformed_value

        return data

    def random_vertical_offset(self, transform_config: dict,
                               data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        rng = self.rng if self.deterministic else np.random

        min, max = transform_config["min"], transform_config["max"]

        offset = rng.uniform(low=min, high=max)

        for channel, value in data.items():
            if channel.value in transform_config["apply_to"]:
                if channel is ChannelEnum.REL_POSITION:
                    transformed_value = value.clone()
                    transformed_value[2] = offset + transformed_value[2]
                else:
                    transformed_value = offset + value

                data[channel] = transformed_value

        return data

    def random_occlusion(self, transform_config: dict,
                         data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        rng = self.rng if self.deterministic else np.random

        probability = transform_config["probability"]

        occlusion = None
        for channel, value in data.items():
            if channel.value in transform_config["apply_to"]:
                if occlusion is None:
                    occlusion = rng.choice([0, 1], p=[1 - probability, probability], size=tuple(value.size()))
                    occlusion = torch.tensor(occlusion)

                if channel == ChannelEnum.OCC_MASK:
                    # assert value.dtype == torch.bool
                    #
                    # transformed_value = value.to(dtype=torch.int) + occlusion
                    #
                    # transformed_value[transformed_value > 1] = 1
                    #
                    # data[channel] = transformed_value.to(dtype=torch.bool)

                    value[occlusion == 1] = True
                elif channel in [ChannelEnum.OCC_DEM, ChannelEnum.OCC_DATA_UM]:
                    value[occlusion == 1] = np.nan
                else:
                    raise NotImplementedError

                data[channel] = value

        return data

    def random_occlusion_dilation(self, transform_config: dict,
                                  data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        rng = self.rng if self.deterministic else np.random

        mask = data[ChannelEnum.OCC_MASK]

        if mask.dtype == torch.bool:
            mask = mask.to(dtype=torch.float)

        if mask.dim() == 2:
            # we need a channel composition with minibatches for 2d convolutions
            mask = mask.unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            raise NotImplementedError

        kernel_size = int(transform_config["kernel_size"])
        # probability to set kernel cell to one
        probability = transform_config["probability"]
        kernel_choice = rng.choice([0, 1], p=[1 - probability, probability], size=(kernel_size, kernel_size))
        # we always set the center of the kernel to 1
        kernel_choice[int(kernel_size // 2), int(kernel_size // 2)] = 1
        kernel = torch.tensor(kernel_choice, dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=0)

        with torch.no_grad():
            dilated_mask = F.conv2d(mask, weight=kernel, padding=1) > 0

        dilated_occlusion = dilated_mask.squeeze(dim=0).squeeze(dim=0)

        data[ChannelEnum.OCC_MASK] = dilated_occlusion

        if ChannelEnum.OCC_DEM in data:
            data[ChannelEnum.OCC_DEM][dilated_occlusion == 1] = np.nan
        if ChannelEnum.OCC_DATA_UM in data:
            data[ChannelEnum.OCC_DATA_UM][dilated_occlusion == 1] = np.nan

        return data

    def random_occlusion_raycasting(self, transform_config: dict,
                               data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        rng = self.rng if self.deterministic else np.random

        if self.grid_map_raycasting is None:
            # we raise the ImportError for grid_map_raycasting on purpose as its required and not available
            import grid_map_raycasting

        occ_mask = data[ChannelEnum.OCC_MASK]
        gt_dem = data[ChannelEnum.GT_DEM]
        res_grid = data[ChannelEnum.RES_GRID]

        vantage_point_u = 0
        vantage_point_v = 0
        vantage_point_elevation = np.NaN
        while np.isnan(vantage_point_elevation):
            # we only accept coordinates where we have elevation information
            vantage_point_u = rng.uniform(low=0, high=gt_dem.size(0))
            vantage_point_v = rng.uniform(low=0, high=gt_dem.size(1))

            vantage_point_elevation = gt_dem[vantage_point_u, vantage_point_v]

        height_viewpoint = rng.uniform(low=transform_config["height_viewpoint"]["min"],
                                       high=transform_config["height_viewpoint"]["high"])

        vantage_point_x = (-gt_dem.size(0) / 2 + vantage_point_u) * res_grid[0]
        vantage_point_y = (-gt_dem.size(1) / 2 + vantage_point_v) * res_grid[1]
        vantage_point_z = vantage_point_elevation + height_viewpoint
        
        vantage_point = np.array([vantage_point_x, vantage_point_y, vantage_point_z], dtype=np.double)

        np_gt_dem = gt_dem.detach().numpy().to(dtype=np.double)

        np_raycasted_occ_mask = self.grid_map_raycasting.rayCastGridMap(vantage_point, np_gt_dem, res_grid)

        data[ChannelEnum.OCC_MASK] = torch.logical_or(occ_mask, torch.tensor(np_raycasted_occ_mask, dtype=torch.bool))

        if ChannelEnum.OCC_DEM in data:
            data[ChannelEnum.OCC_DEM][data[ChannelEnum.OCC_MASK] == 1] = np.nan
        if ChannelEnum.OCC_DATA_UM in data:
            data[ChannelEnum.OCC_DATA_UM][data[ChannelEnum.OCC_MASK] == 1] = np.nan

        return data

    def range_data_uncertainty(self, transform_config: dict,
                               data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        res_grid = data[ChannelEnum.RES_GRID]
        robot_position = data[ChannelEnum.REL_POSITION]

        sample_grid = data[ChannelEnum.OCC_DATA_UM]

        # distance of every pixel from the robot
        lin_x = np.arange(start=-sample_grid.shape[0] / 2, stop=sample_grid.shape[0] / 2, step=1) * res_grid[0].numpy()
        lin_y = np.arange(start=-sample_grid.shape[1] / 2, stop=sample_grid.shape[1] / 2, step=1) * res_grid[1].numpy()
        off_y, off_x = np.meshgrid(lin_x, lin_y)

        dist_x = off_x - robot_position[0].numpy()
        dist_y = off_y - robot_position[1].numpy()
        dist_p2_norm = np.sqrt(np.square(dist_x) + np.square(dist_y))

        stdev = transform_config["stdev"]
        range = transform_config["range"]

        range_data_variance = stdev**2 * np.square(1 / range * dist_p2_norm)

        for channel, value in data.items():
            if channel.value in transform_config["apply_to"]:
                transformed_value = value + value.new_tensor(range_data_variance, dtype=value.dtype)

                data[channel] = transformed_value

        return data
