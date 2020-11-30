from abc import ABC, abstractmethod
import h5py
import json
import numpy as np
import random
import pathlib
import torch
from typing import *

from src.utils.log import create_base_logger, create_logdir


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

        self.min = None
        self.max = None

    def __enter__(self):
        self.hdf5_file = h5py.File(str(self.hdf5_path), 'a')
        self.hdf5_file.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hdf5_file.__exit__()

    def reset(self):
        self.reset_metadata()

    @abstractmethod
    def run(self):
        pass

    def save_to_dataset(self):
        pass

    @staticmethod
    def extend_dataset(dataset: h5py.Dataset, data: Union[np.array, List]):
        if type(data) is list:
            data = np.array(data)

        dataset.resize((dataset.shape[0] + data.shape[0]), axis=0)
        dataset[-data.shape[0]:] = data

    def update_dataset_range(self, elevation_map: np.array):
        # update min and max
        sample_min = np.min(elevation_map[~np.isnan(elevation_map)]).item()
        sample_max = np.max(elevation_map[~np.isnan(elevation_map)]).item()
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

    def reset_metadata(self):
        self.min = None
        self.max = None
