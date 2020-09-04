import h5py
import pathlib
import torch
from typing import *

from .base_dataset import BaseDataset


class OfflineHdf5Dataset(BaseDataset):
    def __init__(self, hdf5_dataset_path: pathlib.Path, **kwargs):
        super().__init__(**kwargs)
        self.hdf5_dataset_path = hdf5_dataset_path

        self.elevation_map_dataset: Optional[h5py.Dataset] = None
        self.occluded_elevation_map_dataset: Optional[h5py.Dataset] = None

        with h5py.File(self.hdf5_dataset_path, 'r') as hdf5_file:
            self.dataset_length = len(hdf5_file[f"/{self.purpose}/elevation_map"])

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        self.set_hdf5_dataset()

        return self.prepare_item(self.elevation_map_dataset[idx, :], self.occluded_elevation_map_dataset[idx, :])

    def __len__(self):
        return self.dataset_length

    def set_hdf5_dataset(self):
        if self.elevation_map_dataset is None or self.occluded_elevation_map_dataset is None:
            hdf5_file = h5py.File(str(self.hdf5_dataset_path), 'r')

            if self.elevation_map_dataset is None:
                self.elevation_map_dataset = hdf5_file[f"/{self.purpose}/elevation_map"]

            if self.occluded_elevation_map_dataset is None:
                self.occluded_elevation_map_dataset = hdf5_file[f"/{self.purpose}/occluded_elevation_map"]
