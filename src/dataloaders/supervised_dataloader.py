import pathlib
from typing import Dict, Union, Sequence, Tuple

import torchvision
from torch.utils.data import DataLoader, Dataset

from .base_dataloader import BaseDataloader

from ..utils.log import get_logger

logger = get_logger("supervised_dataloader")

def get_cifar10_dataset(datadir: pathlib.Path) -> Tuple[Dataset, Dataset]:
    train_data = torchvision.datasets.CIFAR10(str(datadir / "cifar10"), train=True, download=True)
    test_data = torchvision.datasets.CIFAR10(str(datadir / "cifar10"), train=False, download=True)
    return train_data, test_data


class SupervisedDataloader(BaseDataloader):
    def __init__(self, uid, datadir: pathlib.Path, task_logdir: pathlib.Path, **kwargs):
        super().__init__()

        self.config = kwargs

        self.uid = uid

        self.datadir = datadir
        self.logdir = task_logdir / "domains" / f"supervised_dataloader_{self.uid}"
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.base_dataset = kwargs["dataset"]

        self.dataloaders = self.pick_dataloader(self.config)

    def __str__(self):
        return self.uid


    def pick_dataloader(self, domain_config: Dict) -> Dict[str, DataLoader]:
        train_data, test_data = get_cifar10_dataset(self.datadir)

        dataloaders = {}
        for purpose in ["train", "val", "test"]:
            dataloaders[purpose] = DataLoader(train_data,
                                              batch_size=domain_config["batch_size"],
                                              shuffle=domain_config["shuffle"],
                                              num_workers=domain_config["num_workers"])
        return dataloaders
