import h5py
import pathlib
import torch
from typing import *

from .base_dataset import BaseDataset
from src.enums import *


class Hdf5Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.channels = [ChannelEnum.PARAMS, ChannelEnum.ELEVATION_MAP, ChannelEnum.BINARY_OCCLUSION_MAP,
                         ChannelEnum.OCCLUDED_ELEVATION_MAP]

        self.hdf5_dataset_set: bool = False
        self.hdf5_datasets = {}

        with h5py.File(self.dataset_path, 'r') as hdf5_file:
            self.dataset_length = len(hdf5_file[f"/{self.purpose}/{ChannelEnum.ELEVATION_MAP.value}"])

    def __getitem__(self, idx) -> Dict[Union[str, ChannelEnum], torch.Tensor]:
        self.set_hdf5_dataset()

        data = {}
        for channel, hdf5_dataset in self.hdf5_datasets.items():
            data[channel] = hdf5_dataset[idx, ...]

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

            self.hdf5_dataset_set = True
