import numpy as np
from progress.bar import Bar
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
        if hasattr(dataloader.dataset, "dataset_path"):
            logger.info(f"We need to infer the min and max values of the dataset manually for dataset in path "
                        f"{dataloader.dataset.dataset_path}")
        else:
            logger.info(f"We need to infer the min and max values of the dataset manually for dataset")

        length = 0
        min = np.Inf
        max = -np.Inf

        progress_bar = Bar(f"We need to infer the min and max values of the dataset manually", max=len(dataloader))
        for batch_data in dataloader:
            if ChannelEnum.GT_DEM in batch_data:
                sample_data = batch_data[ChannelEnum.GT_DEM]
            elif ChannelEnum.OCC_DEM in batch_data:
                sample_data = batch_data[ChannelEnum.OCC_DEM]
            else:
                raise ValueError

            sample_notnan = sample_data[~torch.isnan(sample_data)]
            sample_min = torch.min(sample_notnan).item()
            sample_max = torch.max(sample_notnan).item()

            length += int(sample_data.size(0))
            min = np.min([min, sample_min]).item()
            max = np.max([max, sample_max]).item()

            progress_bar.next()
        progress_bar.finish()

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
