import h5py
import pathlib
import numpy as np
import torch
from typing import *

from .base_dataset import BaseDataset
from src.enums import *


class AnymalRosbagDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        import rosbag
        self.rosbag_module = rosbag
        self.rosbag_topics = ['/elevation_mapping/elevation_map_recordable']

        self.rosbag = self.rosbag_module.Bag(self.dataset_path, 'r')
        self.dataset_length += self.rosbag.get_message_count(topic_filters=self.rosbag_topics)

    # def __getitem__(self, idx) -> Dict[Union[str, ChannelEnum], torch.Tensor]:
    #     data = {}
    #     for channel, hdf5_dataset in self.hdf5_datasets.items():
    #         data[channel] = hdf5_dataset[idx, ...]
    #
    #     data = self.prepare_keys(data)
    #
    #     return self.prepare_item(data)

    def __iter__(self):
        return iter(range(0, 100))

    def __len__(self):
        return self.dataset_length


