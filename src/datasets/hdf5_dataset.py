import h5py
import pathlib
import torch
from typing import *

from .base_dataset import BaseDataset
from src.enums import *


class Hdf5Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.params_dataset: Optional[h5py.Dataset] = None
        self.elevation_map_dataset: Optional[h5py.Dataset] = None
        self.occluded_elevation_map_dataset: Optional[h5py.Dataset] = None

        with h5py.File(self.dataset_path, 'r') as hdf5_file:
            self.dataset_length = len(hdf5_file[f"/{self.purpose}/elevation_map"])

    def __getitem__(self, idx) -> Dict[Union[str, ChannelEnum], torch.Tensor]:
        self.set_hdf5_dataset()

        data = {ChannelEnum.PARAMS: self.params_dataset[idx, ...],
                ChannelEnum.ELEVATION_MAP: self.elevation_map_dataset[idx, :],
                ChannelEnum.OCCLUDED_ELEVATION_MAP: self.occluded_elevation_map_dataset[idx, :]}

        return self.prepare_item(data)

    def __len__(self):
        return self.dataset_length

    def set_hdf5_dataset(self):
        if self.elevation_map_dataset is None or self.occluded_elevation_map_dataset is None:
            hdf5_file = h5py.File(str(self.dataset_path), 'r')

            if self.params_dataset is None:
                self.params_dataset = hdf5_file[f"/{self.purpose}/params"]

            if self.elevation_map_dataset is None:
                self.elevation_map_dataset = hdf5_file[f"/{self.purpose}/elevation_map"]

            if self.occluded_elevation_map_dataset is None:
                self.occluded_elevation_map_dataset = hdf5_file[f"/{self.purpose}/occluded_elevation_map"]
