from abc import ABC, abstractmethod
import h5py
import numpy as np
import random
import pathlib
import torch
from typing import *

from src.utils.log import create_base_logger, create_logdir


class BaseDatasetGenerator(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs

        self.type = self.config["type"]

        self.logdir = create_logdir(f"dataset_generation_{self.type}")
        self.logger = create_base_logger(self.logdir)

        # set seed
        seeds = self.config.get("seeds", [101])
        assert len(seeds) > 0
        self.seed = seeds[0]
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.hdf5_path = pathlib.Path(self.logdir / "DATASET_OCCLUSION.hdf5")
        self.hdf5_file: Optional[h5py.File] = None

    def __enter__(self):
        self.hdf5_file = h5py.File(str(self.hdf5_path), 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hdf5_file.__exit__()

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