import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import *

from ..utils.log import get_logger
from src.datasets.base_dataset import BaseDataset
from src.enums import *

logger = get_logger("meta_data_info")


class DataloaderMetaInfo:
    def __init__(self, dataloader: DataLoader):
        self.length = len(dataloader.dataset)

        info = recursively_gather_meta_info(dataloader.dataset)

        if info is None:
            self.infer_meta_info(dataloader)
        else:
            self.min = info["min"]
            self.max = info["max"]

    def infer_meta_info(self, dataloader: DataLoader):
        logger.info(f"We need to infer the min and max values of the dataset manually")

        length = 0
        min = np.Inf
        max = -np.Inf

        for batch_data in dataloader:
            if ChannelEnum.GROUND_TRUTH_ELEVATION_MAP in batch_data:
                sample_data = batch_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP]
            elif ChannelEnum.OCCLUDED_ELEVATION_MAP in batch_data:
                sample_data = batch_data[ChannelEnum.OCCLUDED_ELEVATION_MAP]
            else:
                raise ValueError

            sample_notnan = sample_data[~torch.isnan(sample_data)]
            sample_min = torch.min(sample_notnan).item()
            sample_max = torch.max(sample_notnan).item()

            length += int(sample_data.size(0))
            min = np.min([min, sample_min]).item()
            max = np.max([max, sample_max]).item()

        self.length = length
        self.min = min
        self.max = max


def recursively_gather_meta_info(dataset: Dataset) -> Optional[Dict]:
    if issubclass(type(dataset), BaseDataset):
        if dataset.min is None or dataset.max is None:
            return None
        info = {"min": dataset.min, "max": dataset.max}
    elif issubclass(type(dataset), ConcatDataset):
        min = np.Inf
        max = -np.Inf
        for dataset in dataset.datasets:
            dataset_info = recursively_gather_meta_info(dataset)

            if dataset_info is None:
                return None

            min = np.min([min, dataset_info["min"]]).item()
            max = np.max([max, dataset_info["max"]]).item()
        info = {"min": min, "max": max}
    else:
        raise ValueError

    return info
