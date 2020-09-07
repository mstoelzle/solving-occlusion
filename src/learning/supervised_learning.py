import torch

from .base_learning import BaseLearning
from src.enums.channel_enum import ChannelEnum
from src.learning.loss.loss import Loss
from src.learning.tasks import Task
from ..utils.log import get_logger

logger = get_logger("supervised_learning")


class SupervisedLearning(BaseLearning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, task: Task):
        self.set_task(task)
        self.task.loss = Loss(self.task.logdir, **self.task.config["loss"])

        self.set_model(task.model_to_train)
        return self.train_epoches()

    def train_epoch(self, epoch) -> None:
        self.model.train()
        with self.task.loss.new_epoch(epoch, "train"):
            dataloader = self.task.labeled_dataloader.dataloaders['train']

            for batch_idx, data in enumerate(dataloader):
                self.optimizer.zero_grad()

                for key, value in data.items():
                    data[key] = value.to(self.device)
                batch_size = data[ChannelEnum.ELEVATION_MAP].size(0)

                output = self.model(data)

                loss_dict = self.model.loss_function(config=self.task.config["loss"],
                                                     output=output,
                                                     data=data,
                                                     dataset_length=len(dataloader.dataset))
                loss = loss_dict["loss"]
                self.task.loss(batch_size=batch_size, loss_dict=loss_dict)

                loss.backward()
                self.optimizer.step()

    def validate_epoch(self, epoch: int):
        self.model.eval()
        with self.task.loss.new_epoch(epoch, "val"), torch.no_grad():
            dataloader = self.task.labeled_dataloader.dataloaders['val']
            for batch_idx, data in enumerate(dataloader):
                for key, value in data.items():
                    data[key] = value.to(self.device)
                batch_size = data[ChannelEnum.ELEVATION_MAP].size(0)

                output = self.model(data)

                loss_dict = self.model.loss_function(config=self.task.config["loss"],
                                                     output=output,
                                                     data=data,
                                                     dataset_length=len(dataloader.dataset))
                loss = loss_dict["loss"]
                self.task.loss(batch_size=batch_size, loss_dict=loss_dict)

        self.controller.add_state(epoch, self.task.loss.get_epoch_loss(), self.model.state_dict())
