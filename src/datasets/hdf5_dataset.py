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

        self.channels = [ChannelEnum.PARAMS, ChannelEnum.GROUND_TRUTH_ELEVATION_MAP, ChannelEnum.BINARY_OCCLUSION_MAP,
                         ChannelEnum.OCCLUDED_ELEVATION_MAP]

        self.hdf5_dataset_set: bool = False
        self.hdf5_datasets = {}

        with h5py.File(self.dataset_path, 'r') as hdf5_file:
            if f"/{self.purpose}/{ChannelEnum.GROUND_TRUTH_ELEVATION_MAP.value}" in hdf5_file:
                print("sample dataset ground truth")
                sample_dataset = hdf5_file[f"/{self.purpose}/{ChannelEnum.GROUND_TRUTH_ELEVATION_MAP.value}"]
            elif f"/{self.purpose}/elevation_map" in hdf5_file:
                print("sample dataset elevation map")
                sample_dataset = hdf5_file[f"/{self.purpose}/elevation_map"]
            elif f"/{self.purpose}/{ChannelEnum.OCCLUDED_ELEVATION_MAP.value}" in hdf5_file:
                print("sample dataset occluded elevation map ")
                sample_dataset = hdf5_file[f"/{self.purpose}/{ChannelEnum.OCCLUDED_ELEVATION_MAP.value}"]
            else:
                raise ValueError

            self.dataset_length = len(sample_dataset)
            self.min = sample_dataset.attrs.get("min")
            self.max = sample_dataset.attrs.get("max")

            if self.min is None or self.max is None:
                # Attention: this is very memory-demanding
                sample_notnan = sample_dataset[~np.isnan(sample_dataset)]
                self.min = np.min(sample_notnan)
                self.max = np.max(sample_notnan)

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

            for channel in self.channels:
                dataset_url = f"/{self.purpose}/{channel.value}"
                if dataset_url in hdf5_file:
                    self.hdf5_datasets[channel] = hdf5_file[dataset_url]

            if f"/{self.purpose}/elevation_map" in hdf5_file:
                self.hdf5_datasets[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP] = hdf5_file[f"/{self.purpose}/elevation_map"]

            self.hdf5_dataset_set = True
