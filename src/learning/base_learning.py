from abc import ABC, abstractmethod
from copy import deepcopy
import pathlib
from typing import Dict, Optional

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from src.learning.models import pick_model
from src.learning.tasks import Task
from src.learning.controller import Controller
from src.utils.log import get_logger
from src.utils.digest import TensorboardDigest

logger = get_logger("base_learning")


class BaseLearning(ABC):
    def __init__(self, logdir: pathlib.Path, device: torch.device, **kwargs):
        self.task: Optional[Task] = None

        # Is overwritten in set_model_to_device()
        self.device: torch.device = torch.device("cpu:0")

        self.model = None
        self.optimizer = None

        digest_config: Dict = kwargs.get("digest", {})
        if len(digest_config) > 0:
            self.digest = TensorboardDigest(logdir=logdir, **digest_config)
        else:
            self.digest = None

    def reset(self):
        self.task = None
        self.controller = None
        self.model = None
        self.optimizer = None

    def set_task(self, task: Task):
        self.task = task
        self.controller = Controller(**self.task.config.get("controller", {}))

    def set_model(self, model: Optional[torch.nn.Module] = None):
        if model is None or model == "pretrained":
            model_config = deepcopy(self.task.config["model"])

            if model is None:
                logger.warning(f"An untrained model is used for task {self.task.uid}")
                # we need to manually set the use_pretrained parameter to false just for this model config
                model_config["use_pretrained"] = False
            elif model == "pretrained":
                logger.warning(f"A pretrained model is used for task {self.task.uid}")
                # we need to manually set the use_pretrained parameter to true just for this model config
                model_config["use_pretrained"] = True

            model = pick_model(**self.task.config["model"])

        self.model = model.to(self.device)
        self.pick_optimizer()

    def pick_optimizer(self):
        optimizer_config = self.task.config["optimizer"]
        if optimizer_config["name"] == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=optimizer_config["lr"],
                                        weight_decay=optimizer_config.get("weight_decay", 0))
        if optimizer_config["name"] == "SGD":
            self.optimizer = optim.SGD(params=self.model.parameters(),
                                       lr=optimizer_config["lr"],
                                       momentum=optimizer_config["momentum"],
                                       weight_decay=optimizer_config.get("weight_decay", 0))
        else:
            raise NotImplementedError("Pick a valid optimizer")

    @abstractmethod
    def train(self, task: Task):
        pass

    def train_epoches(self):
        self.controller.reset()

        if self.digest is not None:
            self.digest.__enter__()

        logger.info(f"Running {self.task.type} task {self.task.name}")

        self.validate_epoch(-1)  # validate the model once before any training occurs.
        for epoch in self.controller:
            self.train_epoch(epoch)
            self.validate_epoch(epoch)

        best_dict = self.controller.get_best_state()["model_dict"]
        self.model.load_state_dict(best_dict)
        self.task.save_model(self.model)

        self.test()

        if self.digest is not None:
            self.digest.__exit__()

        return self.model

    @abstractmethod
    def train_epoch(self, epoch) -> None:
        pass

    @abstractmethod
    def validate_epoch(self, epoch: int) -> None:
        pass

    def test(self):
        self.model.eval()
        with self.task.loss.new_epoch(0, "test"), torch.no_grad():
            if self.task.type == "supervised-learning":
                dataloader = self.task.labeled_dataloader.dataloaders['test']
            elif self.task.type == "semi-supervised-learning":
                dataloader = self.task.semi_supervised_dataloader.dataloaders['test']
            elif self.task.type == "domain-confusion":
                dataloader = self.task.unlabeled_dataloader.dataloaders['test']
            else:
                raise NotImplementedError(f"The following task type is not implemented: {self.task.type}")

            for batch_idx, data in enumerate(dataloader):
                for key, value in data.items():
                    data[key] = value.to(self.device)
                batch_size = data["occluded_elevation_map"].size(0)

                output = self.model(data["occluded_elevation_map"])

                kld_weight = batch_size / len(dataloader.dataset)
                loss_dict = self.model.loss_function(config=self.task.config["loss"],
                                                     output=output,
                                                     target=data["elevation_map"],
                                                     kld_weight=kld_weight)
                loss = loss_dict["loss"]
                self.task.loss(batch_size=batch_size, loss_dict=loss_dict)
