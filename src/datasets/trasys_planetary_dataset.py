import pathlib
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from typing import *

from .base_dataset import BaseDataset
from src.enums import *


class TrasysPlanetaryDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.samples = []
        self.make_dataset()

    def make_dataset(self):
        for sample_dir in sorted(self.dataset_path.iterdir()):
            if sample_dir.is_dir():
                sample_dict = {}
                for filepath in sorted(sample_dir.iterdir()):
                    if filepath.is_file():
                        filename = filepath.stem
                        if filename == "height":
                            sample_dict[ChannelEnum.GT_DEM] = filepath
                        elif filename == "occlusion":
                            sample_dict[ChannelEnum.OCC_MASK] = filepath
                        else:
                            continue

                self.samples.append(sample_dict)

    def __getitem__(self, idx: int) -> Dict[Union[str, ChannelEnum], torch.Tensor]:
        sample_dict = self.samples[idx]

        data = self.prepare_keys(sample_dict)

        camera_elevation = 2.  # Camera is elevated on 2m

        # TODO: add actual params from dataset metadata
        terrain_resolution = 200. / 128  # 200m terrain length divided by 128 pixels
        # x_grid = data[ChannelEnum.GT_DEM].size(0) // 2
        # y_grid = data[ChannelEnum.GT_DEM].size(1) // 2
        # robot_position_z = data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP][x_grid, y_grid] + camera_elevation

        data[ChannelEnum.REL_POSITION] = torch.tensor([0., 0., camera_elevation])
        data[ChannelEnum.REL_ATTITUDE] = torch.tensor(Rotation.from_euler('zyx', [0, 0, 0]).as_quat())
        data[ChannelEnum.RES_GRID] = torch.tensor([terrain_resolution, terrain_resolution])

        data = self.prepare_item(data, invert_mask=True)

        return data

    def __len__(self):
        return len(self.samples)
