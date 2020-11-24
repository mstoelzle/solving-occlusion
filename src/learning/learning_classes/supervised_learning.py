from progress.bar import Bar
import torch

from src.learning.learning_classes.base_learning import BaseLearning
from src.enums import *
from src.learning.tasks import Task
from src.utils.log import get_logger

logger = get_logger("supervised_learning")


class SupervisedLearning(BaseLearning):
    def __init__(self, **kwargs):
        super().__init__(logger=logger, **kwargs)

    def train(self, task: Task):
        self.set_task(task)

        self.set_model(task.model_to_train)
        return self.train_epochs()

    def train_epoch(self, epoch) -> None:
        self.model.train()

        dataloader = self.task.labeled_dataloader.dataloaders['train']
        with self.task.loss.new_epoch(epoch, "train", dataset=dataloader.dataset):
            progress_bar = Bar(f"Train epoch {epoch} of task {self.task.uid}", max=len(dataloader))
            for batch_idx, data in enumerate(dataloader):
                self.optimizer.zero_grad()

                data = self.dict_to_device(data)
                batch_size = data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP].size(0)

                output = self.model(data)

                loss_dict = self.model.loss_function(loss_config=self.task.config["loss"],
                                                     output=output,
                                                     data=data,
                                                     dataset=dataloader.dataset)
                loss = loss_dict[LossEnum.LOSS]
                self.task.loss(batch_size=batch_size, loss_dict=loss_dict)

                loss.backward()
                self.optimizer.step()
                progress_bar.next()
            progress_bar.finish()

    def validate_epoch(self, epoch: int):
        self.model.eval()

        dataloader = self.task.labeled_dataloader.dataloaders['val']
        with self.task.loss.new_epoch(epoch, "val", dataset=dataloader.dataset), torch.no_grad():
            progress_bar = Bar(f"Validate epoch {epoch} of task {self.task.uid}", max=len(dataloader))
            for batch_idx, data in enumerate(dataloader):
                data = self.dict_to_device(data)
                batch_size = data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP].size(0)

                output = self.model(data)

                loss_dict = self.model.loss_function(loss_config=self.task.config["loss"],
                                                     output=output,
                                                     data=data,
                                                     dataset=dataloader.dataset)
                loss = loss_dict[LossEnum.LOSS]
                self.task.loss(batch_size=batch_size, loss_dict=loss_dict)
                progress_bar.next()
            progress_bar.finish()

        self.controller.add_state(epoch, self.task.loss.get_epoch_loss(), self.model.state_dict())
