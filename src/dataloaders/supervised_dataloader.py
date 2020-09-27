import pathlib
import torch
from torch.utils.data import DataLoader, Dataset
from typing import *

from .base_dataloader import BaseDataloader
from ..utils.log import get_logger
from src.enums import *
from src.datasets import DATASETS

logger = get_logger("supervised_dataloader")


class SupervisedDataloader(BaseDataloader):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

        dataset_config = self.config["dataset"]
        dataset_type = DatasetEnum(dataset_config["type"])

        self.dataloaders = {}
        for purpose in ["train", "val", "test"]:
            dataset = DATASETS[dataset_type](purpose=purpose, dataset_path=pathlib.Path(dataset_config["path"]))

            self.dataloaders[purpose] = DataLoader(dataset=dataset,
                                                   batch_size=self.config["batch_size"],
                                                   shuffle=self.config["shuffle"],
                                                   num_workers=self.config["num_workers"])

    def __str__(self):
        return str(self.config)
