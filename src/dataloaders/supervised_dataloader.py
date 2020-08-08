import pathlib
from typing import Dict, Union, Sequence

import torchvision
from torch.utils.data import DataLoader

from .base_dataloader import BaseDataloader
from .base_dataloader import split_dataloader

from ..tests import test_init_kwargs
from ..utils.log import get_logger


logger = get_logger("supervised_dataloader")


DATALOADER_CREATORS = {
    "CIFAR10": get_cifar10_dataloaders
}


class SupervisedDataloader(BaseDataloader):
    @test_init_kwargs
    def __init__(self, uid, task_logdir: pathlib.Path, **kwargs):
        super().__init__()

        self.config = kwargs

        self.uid = uid

        self.logdir = task_logdir / "domains" / f"supervised_dataloader_{self.uid}"
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.base_dataset = kwargs["dataset"]

        self.dataloaders = pick_dataloader(self.config)

        if "save_samples" in kwargs and kwargs["save_samples"]:
            self.save_samples(n_batches=1)

    def __str__(self):
        return self.uid

    # TODO: this needs more attention
    def save_samples(self, n_batches=0):
        logger.info(f"Saving {n_batches} of sample images for double checking and sanity checks.")
        count = 0

        for (data, target) in self.dataloaders["train"]:
            if len(data.size()) > 2:
                # first we just save a single image
                torchvision.utils.save_image(data[0, ...], self.logdir / f"sample_image_{count}.png")

                # then we save multiple samples in one image
                torchvision.utils.save_image(data, self.logdir / f"sample_images_{count}.png", nrow=8, padding=2,
                                             normalize=False, range=None, scale_each=False,
                                             pad_value=0)
                count += 1
                if count >= n_batches:
                    break
            else:
                plot_toy_dataset(self.logdir, self.labeled_dataloader.dataloaders["train"])
                break


def pick_dataloader(domain_config: Union[dict, Sequence[dict]]) -> Dict[str, DataLoader]:
    if isinstance(domain_config, dict):
        primary_domain_config = domain_config
    else:
        primary_domain_config = domain_config[0]

    dataset_name: str = primary_domain_config["dataset"]

    try:
        dataloader_creator = DATALOADER_CREATORS[dataset_name]
    except KeyError:
        raise ValueError(f"Invalid dataset {dataset_name} selected"
                         f" must be one of: {DATALOADER_CREATORS.keys()}")

    train_loader, valid_loader, test_loader = dataloader_creator(domain_config)

    if "selected_train_split" in primary_domain_config and primary_domain_config.get("total_train_split", 1) > 1:
        total_train_split = primary_domain_config["total_train_split"]
        train_fractions = [1 / float(total_train_split) for i in range(total_train_split)]
        train_loaders = split_dataloader(train_loader, train_fractions, shuffle=True)
        train_loader = train_loaders[primary_domain_config["selected_train_split"]]

    logger.info(f"Created dataloaders for {dataset_name} with lengths: train {len(train_loader)}, "
                f"validation {len(valid_loader)}, test {len(test_loader)}")

    return {"train": train_loader, "val": valid_loader, "test": test_loader}
