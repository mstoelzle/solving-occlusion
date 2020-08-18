import torch

from .base_learning import BaseLearning
from src.learning.loss.supervised_learning_loss import SupervisedLearningLoss
from src.learning.tasks import Task
from ..utils.log import get_logger

logger = get_logger("supervised_learning")


class SupervisedLearning(BaseLearning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, task: Task):
        self.set_task(task)
        self.task.loss = SupervisedLearningLoss(self.task.logdir, **self.task.config["loss"])

        self.set_model(task.model_to_train)
        return self.train_epoches()

    def train_epoch(self, epoch) -> None:
        self.model.train()
        with self.task.loss.new_epoch(epoch, "train"):
            dataloader = self.task.labeled_dataloader.dataloaders['train']

            for batch_idx, (input, target) in enumerate(dataloader):
                self.optimizer.zero_grad()

                input, target = input.to(self.device), target.to(self.device)

                output = self.model(input)

                loss_dict = self.model.loss_function(config=self.task.config["loss"],
                                                     output=output,
                                                     target=target,
                                                     kld_weight=input.size(0) / len(dataloader.dataset))
                loss = loss_dict["loss"]

                if self.digest is not None:
                    self.digest.cache_data(batch_idx, input)

                loss.backward()
                self.optimizer.step()

            if self.digest is not None:
                self.digest.digest(self.model)

    def validate_epoch(self, epoch: int):
        self.model.eval()
        with self.task.loss.new_epoch(epoch, "val"), torch.no_grad():
            dataloader = self.task.labeled_dataloader.dataloaders['val']
            for batch_idx, (input, target) in enumerate(dataloader):
                input, target = input.to(self.device), target.to(self.device)

                output = self.model(input)

                loss_dict = self.model.loss_function(config=self.task.config["loss"],
                                                     output=output,
                                                     target=target,
                                                     kld_weight=input.size(0) / len(dataloader.dataset))
                loss = loss_dict["loss"]

        self.controller.add_state(epoch, self.task.loss.get_epoch_loss(), self.model.state_dict())
