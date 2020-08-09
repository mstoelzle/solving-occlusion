import numpy as np
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from src.utils.log import get_logger

logger = get_logger("base_dataloader")


class BaseDataloader:
    def __init__(self):
        # dataloaders: Dict = {"train": train_loader, "val": valid_loader, "test": test_loader}
        self.dataloaders: Dict = {}


