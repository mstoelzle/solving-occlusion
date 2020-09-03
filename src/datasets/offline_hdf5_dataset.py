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

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        self.set_hdf5_dataset()

        elevation_map = torch.tensor(self.elevation_map_dataset[idx, :]).unsqueeze(dim=0)
        occluded_elevation_map = torch.tensor(self.occluded_elevation_map_dataset[idx, :]).unsqueeze(dim=0)

        return elevation_map, occluded_elevation_map

    def __len__(self):
        self.set_hdf5_dataset()

        return self.elevation_map_dataset.shape[0]

    def set_hdf5_dataset(self):
        if self.elevation_map_dataset is None or self.occluded_elevation_map_dataset is None:
            hdf5_file = h5py.File(str(self.hdf5_dataset_path), 'r')

            if self.elevation_map_dataset is None:
                self.elevation_map_dataset = hdf5_file[f"/{self.purpose}/elevation_map"]

            if self.occluded_elevation_map_dataset is None:
                self.occluded_elevation_map_dataset = hdf5_file[f"/{self.purpose}/occluded_elevation_map"]
