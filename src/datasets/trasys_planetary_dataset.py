import pathlib
import torch
from torchvision.datasets.folder import pil_loader
from typing import *

from .base_dataset import BaseDataset
from src.enums import *


class TrasysPlanetaryDataset(BaseDataset):
    def __init__(self, dataset_path: pathlib.Path, **kwargs):
        super().__init__(**kwargs)

        self.dataset_path = dataset_path
