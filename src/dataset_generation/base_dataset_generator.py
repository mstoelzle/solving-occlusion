from abc import ABC, abstractmethod
import h5py
import json
import numpy as np
import random
import pathlib
import torch
from typing import *

from src.enums import *
from src.utils.log import create_base_logger, create_logdir
from src.visualization.sample_plotter import draw_dataset_samples


class BaseDatasetGenerator(ABC):
    def __init__(self, name: str, remote: bool = False, **kwargs):
        self.config = kwargs

        self.name = name
        self.type = self.config["type"]
        self.remote = remote

        self.logdir = create_logdir(f"dataset_generation_{self.name}")
        self.logger = create_base_logger(self.logdir)

        with open(str(self.logdir / "config.json"), "w") as fp:
            json.dump(self.config, fp, indent=4)

        # set seed
        seeds = self.config.get("seeds", [101])
        assert len(seeds) > 0
        self.seed = seeds[0]
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.hdf5_path = pathlib.Path(self.logdir / "DATASET_OCCLUSION.hdf5")
        self.hdf5_file: Optional[h5py.File] = None
        self.hdf5_group: Optional[h5py.Group] = None
        self.initialized_datasets = False

        self.min = None
        self.max = None

    def __enter__(self):
        self.hdf5_file = h5py.File(str(self.hdf5_path), 'a')
        self.hdf5_file.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hdf5_file.__exit__()

    def reset(self):
        self.reset_metadata()

        self.purpose = None

        self.hdf5_group = None
        self.initialized_datasets = False

        self.sample_idx = 0
        self.total_num_samples = None

    def reset_metadata(self):
        self.min = None
        self.max = None

    def reset_cache(self):
        self.res_grid = []
        self.rel_positions = []
        self.rel_attitudes = []

        self.gt_dems = []
        self.occ_dems = []
        self.occ_masks = []

    @abstractmethod
    def run(self):
        pass

    def create_base_datasets(self, hdf5_group: h5py.Group, max_num_samples: int):
        assert len(self.res_grid) > 0 and len(self.rel_positions) > 0 and len(self.rel_attitudes) > 0

        hdf5_group.create_dataset(name=ChannelEnum.RES_GRID.value,
                                  shape=(0, self.res_grid[0].shape[0]),
                                  maxshape=(max_num_samples, self.res_grid[0].shape[0]))

        hdf5_group.create_dataset(name=ChannelEnum.REL_POSITION.value,
                                  shape=(0, self.rel_positions[0].shape[0]),
                                  maxshape=(max_num_samples, self.rel_positions[0].shape[0]))
        hdf5_group.create_dataset(name=ChannelEnum.REL_ATTITUDE.value,
                                  shape=(0, self.rel_attitudes[0].shape[0]),
                                  maxshape=(max_num_samples, self.rel_attitudes[0].shape[0]))

        self.initialized_datasets = True

    @staticmethod
    def extend_dataset(dataset: h5py.Dataset, data: Union[np.array, List]):
        if type(data) is list:
            data = np.array(data)

        dataset.resize((dataset.shape[0] + data.shape[0]), axis=0)
        dataset[-data.shape[0]:] = data

    def save_cache(self):
        self.extend_dataset(self.hdf5_group[ChannelEnum.RES_GRID.value], self.res_grid)
        self.extend_dataset(self.hdf5_group[ChannelEnum.REL_POSITION.value], self.rel_positions)
        self.extend_dataset(self.hdf5_group[ChannelEnum.REL_ATTITUDE.value], self.rel_attitudes)

        self.reset_cache()

    def update_dataset_range(self, dem: np.array):
        # update min and max
        sample_min = np.min(dem[~np.isnan(dem)]).item()
        sample_max = np.max(dem[~np.isnan(dem)]).item()
        if self.min is None:
            self.min = sample_min
        else:
            self.min = min(self.min, sample_min)

        if self.max is None:
            self.max = sample_max
        else:
            self.max = max(self.max, sample_max)

    def write_metadata(self, purpose_group: h5py.Group):
        for name, dataset in purpose_group.items():
            if isinstance(dataset, h5py.Dataset):
                dataset.attrs["min"] = self.min
                dataset.attrs["max"] = self.max

    def visualize(self, sample_idx: int, res_grid: np.array, rel_position: np.array = None,
                  gt_dem: np.array = None, occ_dem: np.array = None, occ_mask: np.array = None):
        if self.config.get("visualization", None) is not None:
            if self.config["visualization"] is True \
                    or sample_idx % self.config["visualization"].get("frequency", 100) == 0:
                robot_position_pixel = None

                if gt_dem is not None:
                    h, w = gt_dem.shape[0], gt_dem.shape[1]
                else:
                    h, w = occ_dem.shape[0], occ_dem.shape[1]

                if rel_position is not None:
                    u = int(h / 2 + rel_position[0] / res_grid[0])
                    v = int(w / 2 + rel_position[1] / res_grid[1])
                    # we only visualize the robot position if its inside the elevation map
                    plot_robot_position = 0 < u < h and 0 < v < w
                    if plot_robot_position:
                        robot_position_pixel = np.array([u, v])

                sample_dir = self.logdir / f"{self.purpose+'_' if self.purpose is not None else ''}samples"
                if not sample_dir.is_dir():
                    sample_dir.mkdir(exist_ok=True, parents=True)
                draw_dataset_samples(sample_idx=sample_idx, logdir=sample_dir,
                                     gt_dem=gt_dem, occ_dem=occ_dem, occ_mask=occ_mask,
                                     robot_position_pixel=robot_position_pixel, remote=self.remote)
