import numpy as np
import pathlib
import torch
from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader, RandomSampler, SequentialSampler, Subset, \
    WeightedRandomSampler
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

        if "datasets" in self.config:
            datasets_config = self.config["datasets"]
        elif "dataset" in self.config:
            datasets_config = [self.config["dataset"]]
        else:
            raise ValueError

        subsets = {}
        sampling_weights = {}
        for purpose in purposes:
            subsets[purpose] = []
            sampling_weights[purpose] = []

        for dataset_config in datasets_config:
            dataset_type = DatasetEnum(dataset_config["type"])

            dataset_sampling_weights = dataset_config.get("sampling_weights", {"train": 1, "val": 1, "test": 1})

            assert all(purpose in dataset_sampling_weights.keys() for purpose in purposes)

            transforms = {}
            transforms_config = dataset_config.get("transforms", {})
            for purpose in purposes:
                purpose_transforms_config = transforms_config.get(purpose, [])
                if len(purpose_transforms_config) > 0:
                    transforms[purpose] = Transformer(purpose, purpose_transforms_config)
                else:
                    transforms[purpose] = None

            if "split" in dataset_config:
                split = dataset_config["split"]

                assert dataset_type != DatasetEnum.HDF5
                assert list(split.keys()) == purposes

                total_split_sum = np.array(list(split.values())).sum().item()
                for purpose in split.keys():
                    split[purpose] /= total_split_sum

                dataset = DATASETS[dataset_type](config=dataset_config,
                                                 dataset_path=pathlib.Path(dataset_config["path"]))

                indices = np.arange(len(dataset))

                if self.config.get("shuffle", True):
                    rng = np.random.RandomState(seed=0)
                    rng.shuffle(indices)

                start_idx = 0
                for purpose in purposes:
                    len_subset = int(split[purpose] * len(dataset))
                    subset_indices = indices[start_idx:start_idx + len_subset]

                    if dataset_sampling_weights[purpose] <= 0:
                        # we do not want to add the dataset if its not going to get sampled
                        start_idx += len_subset
                        continue

                    # we need to separately create a subset dataset because we need to apply purpose-specific transforms
                    if transforms[purpose] is not None:
                        subset_dataset = DATASETS[dataset_type](config=dataset_config,
                                                                dataset_path=pathlib.Path(dataset_config["path"]),
                                                                purpose=purpose, transform=transforms[purpose])
                        subset = Subset(subset_dataset, subset_indices)
                    else:
                        subset = Subset(dataset, subset_indices)

                    subsets[purpose].append(subset)
                    sampling_weights[purpose].append(dataset_sampling_weights[purpose])

                    start_idx += len_subset
            else:
                for purpose in purposes:
                    if dataset_sampling_weights[purpose] <= 0:
                        # we do not want to add the dataset if its not going to get sampled
                        continue

                    subset_dataset = DATASETS[dataset_type](config=dataset_config,
                                                            dataset_path=pathlib.Path(dataset_config["path"]),
                                                            purpose=purpose, transform=transforms[purpose])
                    subsets[purpose].append(subset_dataset)
                    sampling_weights[purpose].append(dataset_sampling_weights[purpose])

        self.dataloaders = {}
        for purpose in purposes:
            if len(subsets[purpose]) > 1:
                purpose_dataset = ConcatDataset(datasets=subsets[purpose])
            elif len(subsets[purpose]) == 1:
                purpose_dataset = subsets[purpose][0]
            else:
                raise ValueError

            shuffle = self.config.get("shuffle", True)
            if purpose in ["test"]:
                shuffle = False

            if shuffle:
                weights = np.ones(shape=(len(purpose_dataset), ))
                start_idx = 0
                for subset, subset_sampling_weight in zip(subsets[purpose], sampling_weights[purpose]):
                    end_idx = start_idx + len(subset)
                    weights[start_idx:end_idx] = subset_sampling_weight
                    start_idx = end_idx

                sampler = WeightedRandomSampler(weights=weights, num_samples=len(purpose_dataset), replacement=False)

            else:
                for dataset_sampling_weight in sampling_weights[purpose]:
                    if dataset_sampling_weight != 1:
                        raise ValueError("Currently we do not support weighted, sequential sampling")

                sampler = SequentialSampler(purpose_dataset)

            self.dataloaders[purpose] = TorchDataLoader(dataset=purpose_dataset,
                                                        batch_size=self.config["batch_size"],
                                                        sampler=sampler,
                                                        num_workers=self.config["num_workers"])

    def __str__(self):
        return str(self.config)
