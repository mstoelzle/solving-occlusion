from .anymal_rosbag_dataset import AnymalRosbagDataset
from .hdf5_dataset import Hdf5Dataset
from .trasys_planetary_dataset import TrasysPlanetaryDataset
from src.enums import *

DATASETS = {DatasetEnum.ANYMAL_ROSBAG: AnymalRosbagDataset,
            DatasetEnum.HDF5: Hdf5Dataset,
            DatasetEnum.TRASYS_PLANETARY: TrasysPlanetaryDataset}
