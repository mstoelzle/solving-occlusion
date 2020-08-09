import json
import pathlib
from typing import Optional, Type

import torch

from src.dataloaders.supervised_dataloader import SupervisedDataloader
from src.learning.loss.base_loss import BaseLoss
from src.utils.log import get_logger

logger = get_logger("task")


class Task:
    def __init__(self, uid: int, logdir: pathlib.Path, **kwargs):
        self.uid: int = uid
        self.type = kwargs["task_type"]
        self.config = kwargs

        self.logdir: pathlib.Path = logdir

        self.loss: Optional[Type[BaseLoss]] = None

        self.config = kwargs
        self.name = json.dumps(self.config)

        self.labeled_dataloader: Optional[SupervisedDataloader] = None
        self.unlabeled_dataloader: Optional[SupervisedDataloader] = None
        self.inference_dataloader: Optional[SupervisedDataloader] = None

        self.model_to_train = None
        self.model_to_infer = None

        self.output_model: Optional[torch.nn.Module] = None

        # self.save_sample_bool = kwargs['domains']["target"]["save_samples"]
        self.save_infos()

    def __str__(self):
        return self.name

    def save_infos(self):
        with open(str(self.logdir / "info.json"), "w") as fp:
            json.dump({"uid": self.uid, "config": self.config}, fp)

    def save_exit_code(self, code: int):
        with open(str(self.logdir / "exit_code.json"), "w") as fp:
            json.dump({"code": code}, fp, indent=4)

    def save_model(self, model):
        model_path = self.logdir / "model.pt"
        logger.warning(f"Saving model in: {model_path}")
        torch.save(model.state_dict(), str(model_path))

    def cleanup(self):
        # cleanup input models
        self.model_to_train = None
        self.model_to_infer = None

        # cleanup dataloaders
        self.labeled_dataloader = None
        self.unlabeled_dataloader = None
        self.inference_dataloader = None
        self.semi_supervised_dataloader = None

        # cleanup loss class
        self.loss = None
