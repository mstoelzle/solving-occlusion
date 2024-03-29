from progress.bar import Bar
import torch

from src.dataloaders.dataloader_meta_info import DataloaderMetaInfo
from src.enums import *
from src.learning.learning_classes.base_learning import BaseLearning
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
        dataloader_meta_info = DataloaderMetaInfo(dataloader)
        with self.task.loss.new_epoch(epoch, "train", dataloader_meta_info=dataloader_meta_info):
            progress_bar = Bar(f"Train epoch {epoch} of task {self.task.uid}", max=len(dataloader))
            for batch_idx, data in enumerate(dataloader):
                self.optimizer.zero_grad()

                data = self.dict_to_device(data)
                batch_size = data[ChannelEnum.GT_DEM].size(0)

                output = self.model.forward_pass(data)

                if torch.isnan(output[ChannelEnum.REC_DEM]).sum() > 0:
                    raise RuntimeError("We detected NaNs in the model outputs which means "
                                       "that the training is diverging")

                loss_dict = self.model.loss_function(loss_config=self.task.config["loss"],
                                                     output=output,
                                                     data=data,
                                                     dataloader_meta_info=dataloader_meta_info,
                                                     feature_extractor=self.feature_extractor)
                loss = loss_dict[LossEnum.LOSS]
                self.task.loss(batch_size=batch_size, loss_dict=loss_dict)

                loss.backward()
                self.optimizer.step()
                progress_bar.next()
            progress_bar.finish()

    def validate_epoch(self, epoch: int):
        self.model.eval()

        dataloader = self.task.labeled_dataloader.dataloaders['val']
        dataloader_meta_info = DataloaderMetaInfo(dataloader)
        with self.task.loss.new_epoch(epoch, "val", dataloader_meta_info=dataloader_meta_info), torch.no_grad():
            progress_bar = Bar(f"Validate epoch {epoch} of task {self.task.uid}", max=len(dataloader))
            for batch_idx, data in enumerate(dataloader):
                data = self.dict_to_device(data)
                batch_size = data[ChannelEnum.GT_DEM].size(0)

                output = self.model.forward_pass(data)

                loss_dict = self.model.loss_function(loss_config=self.task.config["loss"],
                                                     output=output,
                                                     data=data,
                                                     dataloader_meta_info=dataloader_meta_info)
                self.task.loss(batch_size=batch_size, loss_dict=loss_dict)
                progress_bar.next()
            progress_bar.finish()

        self.controller.add_state(epoch, self.task.loss.get_epoch_loss(), self.model.state_dict())
