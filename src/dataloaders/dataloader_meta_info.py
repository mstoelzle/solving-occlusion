import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.log import get_logger
from src.enums import *

logger = get_logger("meta_data_info")


class DataloaderMetaInfo:
    def __init__(self, dataloader: DataLoader):
        if hasattr(dataloader, "datasets"):
            self.purpose = None
            self.length = 0
            min = np.Inf
            max = -np.Inf
            for dataset in dataloader.datasets:
                if self.purpose is None:
                    self.purpose = dataset.purpose
                else:
                    assert self.purpose == dataset.purpose

                self.length += len(dataset)
                min = np.min([min, dataset.min])
                max = np.max([max, dataset.max])

                if dataset.min is None or dataset.max is None:
                    self.infer_meta_info(dataloader)
                    break

            self.min = min
            self.max = max
        elif hasattr(dataloader, "dataset"):
            dataset = dataloader.dataset
            self.purpose = dataset.purpose
            self.length = len(dataset)

            if dataset.min is None or dataset.max is None:
                self.infer_meta_info(dataloader)
            else:
                self.min = dataset.min
                self.max = dataset.max
        else:
            raise ValueError

    def infer_meta_info(self, dataloader: DataLoader):
        logger.info(f"We need to infer the min and max values of the dataset manually for purpose {purpose}")

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
            min = np.min([min, sample_min])
            max = np.max([max, sample_max])

        self.length = length
        self.min = min
        self.max = max