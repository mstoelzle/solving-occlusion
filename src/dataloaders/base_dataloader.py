import numpy as np
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

from .datasets import split_dataset
from path_learning.utils.log import get_logger

logger = get_logger("base_dataloader")


class BaseDataloader:
    def __init__(self):
        # dataloaders: Dict = {"train": train_loader, "val": valid_loader, "test": test_loader}
        self.dataloaders: Dict = {}


def split_dataloader(dataloader: DataLoader, fractions: List, shuffle: bool = True,
                     balance: bool = True) -> List[DataLoader]:
    logger.info(f"Splitting dataset with length {len(dataloader.dataset)} "
                f"into {fractions} with shuffle={shuffle} and balance={balance}")

    subsets = split_dataset(dataloader.dataset, fractions, seed=1, shuffle=shuffle, balance=balance,
                            batch_size=dataloader.batch_size)

    dataloaders = []
    for subset in subsets:
        dataloader = DataLoader(subset, shuffle=True,
                                batch_size=dataloader.batch_size, num_workers=dataloader.num_workers)
        dataloaders.append(dataloader)

    return dataloaders

