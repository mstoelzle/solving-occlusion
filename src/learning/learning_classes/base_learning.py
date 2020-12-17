from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import h5py
import logging
import pathlib
from progress.bar import Bar
from typing import *

import torch
from torch import optim

from src.dataloaders.dataloader_meta_info import DataloaderMetaInfo
from src.enums import *
from src.learning.controller.controller import Controller
from src.learning.loss.loss import Loss
from src.learning.models import pick_model
from src.learning.models.baseline.base_baseline_model import BaseBaselineModel
from src.learning.tasks import Task
from src.utils.digest import TensorboardDigest
from src.utils.log import get_logger, log_memory_usage
from src.utils.runtime import RuntimeStatistics

logger = get_logger("base_learning")


class BaseLearning(ABC):
    def __init__(self, seed: int, logdir: pathlib.Path, device: torch.device, logger: logging.Logger,
                 results_hdf5_path: pathlib.Path, **kwargs):
        super().__init__()

        self.seed = seed
        self.logdir = logdir
        self.logger = logger

        self.task: Optional[Task] = None

        # Is overwritten in set_model_to_device()
        self.device: torch.device = device

        self.model = None
        self.optimizer = None

        digest_config: Dict = kwargs.get("digest", {})
        if len(digest_config) > 0:
            self.digest = TensorboardDigest(logdir=self.logdir, **digest_config)
        else:
            self.digest = None

        self.results_hdf5_path: pathlib.Path = results_hdf5_path
        self.results_hdf5_file: Optional[h5py.File] = None

    def reset(self):
        self.task = None
        self.controller = None
        self.model = None
        self.optimizer = None

    def set_task(self, task: Task):
        self.task = task

        self.controller = Controller(**self.task.config.get("controller", {}))
        self.task.loss = Loss(self.task.logdir, **self.task.config["loss"])

    def set_model(self, model: Union[str, Optional[torch.nn.Module], pathlib.Path] = None,
                  pick_optimizer: bool = True):
        if model is None or model == "pretrained" or issubclass(type(model), pathlib.Path):
            model_spec = model
            model_config = deepcopy(self.task.config["model"])

            model_config["pretrained"] = False
            if model == "pretrained":
                self.logger.info(f"An pretrained model is used for task {self.task.uid}")
                # we need to manually set the use_pretrained parameter to true just for this model config
                model_config["pretrained"] = True

            model = pick_model(seed=self.seed, **self.task.config["model"])

            if issubclass(type(model_spec), pathlib.Path):
                self.logger.info(f"Loading a model for task {self.task.uid} from {str(model_spec)}")
                state_dict = torch.load(str(model_spec), map_location=self.device)

                model = model.to(self.device)
                model.load_state_dict(state_dict)

        self.model = model.to(self.device)

        # we do not need an optimizer for our baseline models with traditional algorithms
        if pick_optimizer is True and not (issubclass(type(self.model), BaseBaselineModel)):
            self.pick_optimizer()
        else:
            self.optimizer = None

    def pick_optimizer(self):
        optimizer_config = self.task.config["optimizer"]
        if optimizer_config["name"] == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=optimizer_config["lr"],
                                        weight_decay=optimizer_config.get("weight_decay", 0))
        elif optimizer_config["name"] == "SGD":
            self.optimizer = optim.SGD(params=self.model.parameters(),
                                       lr=optimizer_config["lr"],
                                       momentum=optimizer_config["momentum"],
                                       weight_decay=optimizer_config.get("weight_decay", 0))
        else:
            raise NotImplementedError("Pick a valid optimizer")

    def __enter__(self):
        self.results_hdf5_file = h5py.File(str(self.results_hdf5_path), 'a')
        self.results_hdf5_file.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # export scalar data to JSON for external processing
        self.results_hdf5_file.__exit__()

    # @abstractmethod
    # def train(self, task: Task):
    #     pass

    def train_epochs(self):
        self.controller.reset()

        self.logger.info(f"Running {self.task.type} task {self.task.name}")

        self.validate_epoch(-1)  # validate the model once before any training occurs.

        save_frequency = self.task.config.get("model", {}).get("save_frequency", None)

        if self.optimizer is not None:
            for epoch in self.controller:
                log_memory_usage(f"before epoch {epoch}", self.logger)
                self.train_epoch(epoch)
                self.validate_epoch(epoch)

                if save_frequency is not None and epoch % save_frequency == 0:
                    best_dict = self.controller.get_best_state()["model_dict"]
                    self.task.save_state_dict(best_dict)

            best_dict = self.controller.get_best_state()["model_dict"]
            self.model.load_state_dict(best_dict)
            self.task.save_state_dict(best_dict)

        self.test()

        return self.model

    # @abstractmethod
    # def train_epoch(self, epoch) -> None:
    #     pass
    #
    # @abstractmethod
    # def validate_epoch(self, epoch: int) -> None:
    #     pass

    def test(self):
        hdf5_group_prefix = f"/task_{self.task.uid}/test"
        test_data_hdf5_group = self.results_hdf5_file.create_group(f"/{hdf5_group_prefix}/data")
        test_loss_hdf5_group = self.results_hdf5_file.create_group(f"/{hdf5_group_prefix}/loss")

        self.model.eval()

        if self.task.type == TaskTypeEnum.SUPERVISED_LEARNING:
            dataloader = self.task.labeled_dataloader.dataloaders['test']
        else:
            raise NotImplementedError(f"The following task type is not implemented: {self.task.type}")

        runtime_statistics = RuntimeStatistics("test", dataloader.batch_size,
                                               log=True, logdir=self.task.logdir)

        dataloader_meta_info = DataloaderMetaInfo(dataloader)
        with self.task.loss.new_epoch(0, "test", dataloader_meta_info=dataloader_meta_info), torch.no_grad():
            start_idx = 0
            progress_bar = Bar(f"Test inference for task {self.task.uid}", max=len(dataloader))
            for batch_idx, data in enumerate(dataloader):
                data = self.dict_to_device(data)
                batch_size = data[ChannelEnum.GT_DEM].size(0)

                with runtime_statistics:
                    output = self.model(data)

                self.add_batch_data_to_hdf5_results(test_data_hdf5_group, data, start_idx,
                                                    dataloader_meta_info.length)
                self.add_batch_data_to_hdf5_results(test_data_hdf5_group, output, start_idx,
                                                    dataloader_meta_info.length)

                loss_dict = self.model.loss_function(loss_config=self.task.config["loss"],
                                                     output=output,
                                                     data=data,
                                                     dataloader_meta_info=dataloader_meta_info,
                                                     reduction="mean_per_sample")
                aggregated_loss_dict = self.task.loss.aggregate_mean_loss_dict(loss_dict)
                self.task.loss(batch_size=batch_size, loss_dict=aggregated_loss_dict)
                self.add_batch_data_to_hdf5_results(test_loss_hdf5_group, loss_dict, start_idx,
                                                    dataloader_meta_info.length)

                start_idx += batch_size
                progress_bar.next()

            duration_mean, duration_std = runtime_statistics.collect_statistics()
            progress_bar.finish()

    def infer(self):
        hdf5_group_prefix = f"/task_{self.task.uid}/inference"
        data_hdf5_group = self.results_hdf5_file.create_group(f"/{hdf5_group_prefix}/data")

        self.model.eval()
        if self.task.type in [TaskTypeEnum.SUPERVISED_LEARNING, TaskTypeEnum.INFERENCE]:
            dataloader = self.task.labeled_dataloader.dataloaders['test']
        else:
            raise NotImplementedError(f"The following task type is not implemented: {self.task.type}")

        dataloader_meta_info = DataloaderMetaInfo(dataloader)

        start_idx = 0
        progress_bar = Bar(f"Inference for task {self.task.uid}", max=len(dataloader))
        for batch_idx, data in enumerate(dataloader):
            data = self.dict_to_device(data)
            batch_size = data[ChannelEnum.OCC_DEM].size(0)

            output = self.model(data)
            self.add_batch_data_to_hdf5_results(data_hdf5_group, data, start_idx, dataloader_meta_info.length)
            self.add_batch_data_to_hdf5_results(data_hdf5_group, output, start_idx, dataloader_meta_info.length)

            start_idx += batch_size
            progress_bar.next()
        progress_bar.finish()

    def dict_to_device(self, data: Dict[Union[ChannelEnum, str], torch.Tensor]) \
            -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        for key, value in data.items():
            data[key] = value.to(self.device)
        return data

    @staticmethod
    def add_batch_data_to_hdf5_results(hdf5_group: h5py.Group, batch_data: dict,
                                       start_idx: int, total_length: int):
        for key, value in batch_data.items():
            if issubclass(type(key), Enum):
                key = key.value

            if type(value) == torch.Tensor:
                value = value.detach().cpu()
                value_shape = list(value.size())
            else:
                value_shape = list(value.shape)

            if key not in hdf5_group:
                max_value_shape = value_shape.copy()
                max_value_shape[0] = total_length
                hdf5_dataset = hdf5_group.create_dataset(key, shape=max_value_shape)
            else:
                hdf5_dataset = hdf5_group[key]

            hdf5_dataset[start_idx:start_idx + value_shape[0], ...] = value
