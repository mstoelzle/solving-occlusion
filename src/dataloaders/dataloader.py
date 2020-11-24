import numpy as np
import pathlib
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Subset
from typing import *

from ..utils.log import get_logger
from src.enums import *
from src.datasets import DATASETS
from src.datasets.transforms import Transformer

logger = get_logger("dataloader")


class Dataloader:
    def __init__(self, purposes=["train", "val", "test"], **kwargs):
        super().__init__()
        self.config = kwargs

        dataset_config = self.config["dataset"]
        dataset_type = DatasetEnum(dataset_config["type"])

        transforms = {}
        transforms_config = dataset_config.get("transforms", {})
        for purpose in ["train", "val", "test"]:
            purpose_transforms_config = transforms_config.get(purpose, [])
            if len(purpose_transforms_config) > 0:
                transforms[purpose] = Transformer(purpose, purpose_transforms_config)
            else:
                transforms[purpose] = None

        subsets = {}
        if "split" in dataset_config:
            split = dataset_config["split"]

            assert dataset_type != DatasetEnum.HDF5
            assert list(split.keys()) == ["train", "val", "test"]

            total_split_sum = np.array(list(split.values())).sum().item()
            for purpose in split.keys():
                split[purpose] /= total_split_sum

            dataset = DATASETS[dataset_type](config=dataset_config, dataset_path=pathlib.Path(dataset_config["path"]))

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
                    subset_dataset = DATASETS[dataset_type](config=dataset_config,
                                                            dataset_path=pathlib.Path(dataset_config["path"]),
                                                            purpose=purpose, transform=transforms[purpose])
                    subsets[purpose] = Subset(subset_dataset, subset_indices)
                else:
                    subsets[purpose] = Subset(dataset, subset_indices)

                start_idx += len_subset

        self.dataloaders = {}
        for purpose in purposes:
            if len(subsets) > 0:
                purpose_dataset = subsets[purpose]
            else:
                purpose_dataset = DATASETS[dataset_type](config=dataset_config,
                                                         dataset_path=pathlib.Path(dataset_config["path"]),
                                                         purpose=purpose, transform=transforms[purpose])

            shuffle = self.config.get("shuffle", True)
            if purpose in ["test"]:
                shuffle = False

            self.dataloaders[purpose] = TorchDataLoader(dataset=purpose_dataset,
                                                        batch_size=self.config["batch_size"],
                                                        shuffle=shuffle,
                                                        num_workers=self.config["num_workers"])

            self.init_meta_data(purpose, self.dataloaders[purpose])

    def __str__(self):
        return str(self.config)

    def init_meta_data(self, purpose: str, dataloader: TorchDataLoader):
        dataset = dataloader.dataset

        if dataset.min is None or dataset.max is None:
            logger.info(f"We need to infer the min and max values of the dataset manually for purpose {purpose}")

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

                min = np.min([min, sample_min])
                max = np.max([max, sample_max])

            dataset.min = min
            dataset.max = max
