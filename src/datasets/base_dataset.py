from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple

from src.utils.log import get_logger

logger = get_logger("base_dataset")


class BaseDataset(ABC):
    def __init__(self, purpose: str):
        self.purpose = purpose

    def preprocess_item(self, elevation_map: np.array, occluded_elevation_map: np.array) -> Dict[str, torch.Tensor]:
        elevation_map = torch.tensor(elevation_map).unsqueeze(dim=0)
        occluded_elevation_map = torch.tensor(occluded_elevation_map).unsqueeze(dim=0)
        output = {"elevation_map": elevation_map, "occluded_elevation_map": occluded_elevation_map}
        return output
