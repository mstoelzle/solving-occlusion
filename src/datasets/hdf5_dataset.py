import h5py
import pathlib
import numpy as np
import torch
from typing import *

from .base_dataset import BaseDataset
from src.enums import *


class Hdf5Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.hdf5_dataset_set: bool = False
        self.hdf5_datasets = {}

        with h5py.File(self.dataset_path, 'r') as hdf5_file:
            if f"/{self.purpose}/{ChannelEnum.GT_DEM.value}" in hdf5_file:
                sample_dataset = hdf5_file[f"/{self.purpose}/{ChannelEnum.GT_DEM.value}"]
            elif f"/{self.purpose}/elevation_map" in hdf5_file:
                sample_dataset = hdf5_file[f"/{self.purpose}/elevation_map"]
            elif f"/{self.purpose}/ground_truth_elevation_map" in hdf5_file:
                sample_dataset = hdf5_file[f"/{self.purpose}/ground_truth_elevation_map"]
            elif f"/{self.purpose}/{ChannelEnum.OCC_DEM.value}" in hdf5_file:
                sample_dataset = hdf5_file[f"/{self.purpose}/{ChannelEnum.OCC_DEM.value}"]
            elif f"/{self.purpose}/occluded_elevation_map" in hdf5_file:
                sample_dataset = hdf5_file[f"/{self.purpose}/occluded_elevation_map"]
            else:
                raise ValueError

            self.dataset_length = len(sample_dataset)
            self.min = sample_dataset.attrs.get("min")
            self.max = sample_dataset.attrs.get("max")

    def __getitem__(self, idx) -> Dict[Union[str, ChannelEnum], torch.Tensor]:
        self.set_hdf5_dataset()

        data = {}
        for channel, hdf5_dataset in self.hdf5_datasets.items():
            data[channel] = hdf5_dataset[idx, ...]

        data = self.prepare_keys(data)

        return self.prepare_item(data)

    def __len__(self):
        return self.dataset_length

    def set_hdf5_dataset(self):
        if self.hdf5_dataset_set is False:
            hdf5_file = h5py.File(str(self.dataset_path), 'r')

            if self.purpose is not None:
                hdf5_group = hdf5_file[self.purpose]
            else:
                hdf5_group = hdf5_file

            for channel_value, hdf5_dataset in hdf5_group.items():
                channel = ChannelEnum(channel_value)
                self.hdf5_datasets[channel] = hdf5_dataset

            self.hdf5_dataset_set = True
