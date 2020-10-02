import h5py
import pathlib
import torch
from typing import *

from .base_dataset import BaseDataset
from src.enums import *


class Hdf5Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.hdf5_dataset_set: bool = False
        self.params_dataset: Optional[h5py.Dataset] = None
        self.elevation_map_dataset: Optional[h5py.Dataset] = None
        self.binary_occlusion_map_dataset: Optional[h5py.Dataset] = None
        self.occluded_elevation_map_dataset: Optional[h5py.Dataset] = None

        with h5py.File(self.dataset_path, 'r') as hdf5_file:
            self.dataset_length = len(hdf5_file[f"/{self.purpose}/elevation_map"])

    def __getitem__(self, idx) -> Dict[Union[str, ChannelEnum], torch.Tensor]:
        self.set_hdf5_dataset()

        data = {}
        if self.params_dataset is not None:
            data[ChannelEnum.PARAMS] = self.params_dataset[idx, ...]
        if self.elevation_map_dataset is not None:
            data[ChannelEnum.ELEVATION_MAP] = self.elevation_map_dataset[idx, :]
        if self.binary_occlusion_map_dataset is not None:
            data[ChannelEnum.BINARY_OCCLUSION_MAP] = self.binary_occlusion_map_dataset[idx, :]
        if self.occluded_elevation_map_dataset is not None:
            data[ChannelEnum.OCCLUDED_ELEVATION_MAP] = self.occluded_elevation_map_dataset[idx, :]

        return self.prepare_item(data)

    def __len__(self):
        return self.dataset_length

    def set_hdf5_dataset(self):
        if self.hdf5_dataset_set is False:
            hdf5_file = h5py.File(str(self.dataset_path), 'r')

            if f"/{self.purpose}/params" in hdf5_file:
                self.params_dataset = hdf5_file[f"/{self.purpose}/params"]

            if f"/{self.purpose}/elevation_map" in hdf5_file:
                self.elevation_map_dataset = hdf5_file[f"/{self.purpose}/elevation_map"]

            if f"/{self.purpose}/binary_occlusion_map" in hdf5_file:
                self.binary_occlusion_map_dataset = hdf5_file[f"/{self.purpose}/binary_occlusion_map"]

            if f"/{self.purpose}/occluded_elevation_map" in hdf5_file:
                self.occluded_elevation_map_dataset = hdf5_file[f"/{self.purpose}/occluded_elevation_map"]

            self.hdf5_dataset_set = True
