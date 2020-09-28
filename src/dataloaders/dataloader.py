import numpy as np
import pathlib
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Subset
from typing import *

from ..utils.log import get_logger
from src.enums import *
from src.datasets import DATASETS
from src.datasets.transforms import assemble_transforms

logger = get_logger("dataloader")


class Dataloader:
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

        dataset_config = self.config["dataset"]
        dataset_type = DatasetEnum(dataset_config["type"])

        transforms = {}
        transforms_config = dataset_config.get("transforms", {})
        for purpose in ["train", "val", "test"]:
            transforms[purpose] = assemble_transforms(purpose, transforms_config.get(purpose, {}),
                                                      dataset_config["size"], True)

        subsets = {}
        if "split" in dataset_config:
            split = dataset_config["split"]

            assert dataset_type != DatasetEnum.HDF5
            assert list(split.keys()) == ["train", "val", "test"]

            total_split_sum = np.array(list(split.values())).sum().item()
            for purpose in split.keys():
                split[purpose] /= total_split_sum

            dataset = DATASETS[dataset_type](dataset_path=pathlib.Path(dataset_config["path"]))

            indices = np.arange(len(dataset))

            if self.config.get("shuffle", True):
                rng = np.random.RandomState(seed=0)
                rng.shuffle(indices)

            start_idx = 0
            for purpose in split.keys():
                len_subset = int(split[purpose] * len(dataset))
                subset_indices = indices[start_idx:start_idx + len_subset]

                # we need to separately create a subset dataset because we need to apply purpose-specific transforms
                if transforms[purpose] is not None:
                    subset_dataset = DATASETS[dataset_type](dataset_path=pathlib.Path(dataset_config["path"]),
                                                            purpose=purpose, transform=transforms[purpose])
                    subsets[purpose] = Subset(subset_dataset, subset_indices)
                else:
                    subsets[purpose] = Subset(dataset, subset_indices)

                start_idx += len_subset

        self.dataloaders = {}
        for purpose in ["train", "val", "test"]:
            if len(subsets) > 0:
                purpose_dataset = subsets[purpose]
            else:
                purpose_dataset = DATASETS[dataset_type](dataset_path=pathlib.Path(dataset_config["path"]),
                                                         purpose=purpose, transform=transforms[purpose])

            self.dataloaders[purpose] = TorchDataLoader(dataset=purpose_dataset,
                                                        batch_size=self.config["batch_size"],
                                                        shuffle=self.config.get("shuffle", True),
                                                        num_workers=self.config["num_workers"])

    def __str__(self):
        return str(self.config)
