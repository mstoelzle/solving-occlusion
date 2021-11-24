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
import torch.autograd.profiler as profiler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, ReduceLROnPlateau, MultiplicativeLR

from src.dataloaders.dataloader_meta_info import DataloaderMetaInfo
from src.enums import *
from src.learning.controller.controller import Controller
from src.learning.loss.loss import Loss
from src.learning.models import pick_model
from src.learning.models.baseline.base_baseline_model import BaseBaselineModel
from src.learning.models.baseline.lsq_plane_fit_baseline import LsqPlaneFitBaseline
from src.learning.models.unet.unet_parts import VGG16FeatureExtractor
from src.learning.tasks import Task
from src.traversability.traversability_assessment import TraversabilityAssessment
from src.utils.log import get_logger

logger = get_logger("base_learning")


class BaseLearning(ABC):
    def __init__(self, seed: int, logdir: pathlib.Path, device: torch.device, logger: logging.Logger,
                 results_hdf5_path: pathlib.Path, remote: bool = False, **kwargs):
        super().__init__()

        self.seed = seed
        self.logdir = logdir
        self.logger = logger
        self.remote = remote

        self.task: Optional[Task] = None

        # Is overwritten in set_model_to_device()
        self.device: torch.device = device

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.results_hdf5_path: pathlib.Path = results_hdf5_path
        self.results_hdf5_file: Optional[h5py.File] = None

        self.feature_extractor = None

    def reset(self):
        self.task = None
        self.controller = None
        self.model = None
        self.optimizer = None
        self.feature_extractor = None

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
            self.pick_scheduler(optimizer=self.optimizer)
        else:
            self.optimizer = None
            self.scheduler = None

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

        return self.optimizer

    def pick_scheduler(self, optimizer: torch.optim.Optimizer):
        SCHEDULER_DICT = {
            "cosine_annealing_warm_restarts": CosineAnnealingWarmRestarts,
            "multiplicative_lr": MultiplicativeLR,
            "reduce_on_plateau": ReduceLROnPlateau,
            "lambda_lr": LambdaLR,
        }

        EXPECTED_KWARGS = {
            "cosine_annealing_warm_restarts": ["T_0", "T_mult", "eta_min"],
            "multiplicative_lr": ["lr_lambda"],
            "reduce_on_plateau": ["factor", "patience"]
        }

        scheduler_config = self.task.config.get("scheduler", None)
        if scheduler_config is None or len(scheduler_config) == 0:
            return

        scheduler_name = scheduler_config["name"]
        scheduler = None
        assert scheduler_name in SCHEDULER_DICT, f"Chosen scheduler name {scheduler_name} is not implemented " \
                                                 f"in scheduler dict {SCHEDULER_DICT}"
        logger.info(f"scheduler_config: {scheduler_config}, expected arguments: {EXPECTED_KWARGS[scheduler_name]}, ")
        for kwarg in EXPECTED_KWARGS[scheduler_name]:
            assert kwarg in scheduler_config.keys(), f"Expected keyword {kwarg} for LR scheduler selection not provided " \
                                           f"in keywords {scheduler_config.keys()}."

        if scheduler_name == "multiplicative_lr":
            logger.info(f'kwargs["lr_lambda"]: {scheduler_config["lr_lambda"]}')
            # Multiplicate LR needs function not value with which to rescale
            scheduler = SCHEDULER_DICT[scheduler_name](optimizer, lr_lambda=lambda epoch: scheduler_config["lr_lambda"])
        elif scheduler_name == "cosine_annealing_warm_restarts":
            scheduler = SCHEDULER_DICT[scheduler_name](optimizer, T_0=scheduler_config["T_0"],
                                                       T_mult=scheduler_config["T_mult"],
                                                       eta_min=scheduler_config["eta_min"])
        assert scheduler is not None, "scheduler was not assigned"

        self.scheduler = scheduler

        return scheduler

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
            if self.model.config.get("feature_extractor", False) is True:
                self.feature_extractor = VGG16FeatureExtractor()
                self.feature_extractor = self.feature_extractor.to(device=self.device)

            for epoch in self.controller:
                self.train_epoch(epoch)
                self.validate_epoch(epoch)

                if save_frequency is not None and epoch % save_frequency == 0:
                    best_dict = self.controller.get_best_state()["model_dict"]
                    self.task.save_state_dict(best_dict)

            best_dict = self.controller.get_best_state()["model_dict"]
            self.model.load_state_dict(best_dict)
            self.task.save_state_dict(best_dict)

        if self.task.config.get("model", {}).get("trace", True) and self.model.strict_forward_def is True:
            self.trace_model()

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

        traversability_assessment = None
        if self.task.config.get("traversability_assessment", {}).get("active", False):
            traversability_config = self.task.config.get("traversability_assessment", {})
            traversability_assessment = TraversabilityAssessment(**traversability_config)

        self.model.eval()

        if self.task.type == TaskTypeEnum.SUPERVISED_LEARNING:
            dataloader = self.task.labeled_dataloader.dataloaders['test']
        else:
            raise NotImplementedError(f"The following task type is not implemented: {self.task.type}")

        dataloader_meta_info = DataloaderMetaInfo(dataloader)
        with self.task.loss.new_epoch(0, "test", dataloader_meta_info=dataloader_meta_info), torch.no_grad():
            prof = None
            if not isinstance(self.model, LsqPlaneFitBaseline):
                prof = profiler.profile()
                prof.__enter__()

            start_idx = 0
            progress_bar = Bar(f"Test inference for task {self.task.uid}", max=len(dataloader))
            for batch_idx, data in enumerate(dataloader):
                data = self.dict_to_device(data)
                batch_size = data[ChannelEnum.GT_DEM].size(0)

                if isinstance(self.model, LsqPlaneFitBaseline):
                    # the profiler somehow has issues with the scipy lsq solver
                    output = self.model.forward_pass(data)
                else:
                    with profiler.record_function("model_inference"):
                        output = self.model.forward_pass(data)

                if traversability_assessment is not None:
                    output = traversability_assessment(output=output, data=data)

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

            progress_bar.finish()

        if not isinstance(self.model, LsqPlaneFitBaseline):
            prof.__exit__(0, None, None)
            with open(str(self.task.logdir / "test_cputime.txt"), "a") as f:
                f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            if self.task.config.get("profiler_export_chrome_trace", False):
                prof.export_chrome_trace(str(self.task.logdir / "test_cputime_chrome_trace.json"))

    def infer(self):
        hdf5_group_prefix = f"/task_{self.task.uid}/inference"
        data_hdf5_group = self.results_hdf5_file.create_group(f"/{hdf5_group_prefix}/data")

        self.model.eval()
        if self.task.type in [TaskTypeEnum.SUPERVISED_LEARNING, TaskTypeEnum.INFERENCE]:
            dataloader = self.task.labeled_dataloader.dataloaders['test']
        else:
            raise NotImplementedError(f"The following task type is not implemented: {self.task.type}")

        dataloader_meta_info = DataloaderMetaInfo(dataloader)

        subgrid_size = self.task.labeled_dataloader.config.get("subgrid_size")

        prof = None
        if not isinstance(self.model, LsqPlaneFitBaseline):
            prof = profiler.profile()
            prof.__enter__()

        start_idx = 0
        progress_bar = Bar(f"Inference for task {self.task.uid}", max=len(dataloader))
        for batch_idx, data in enumerate(dataloader):
            data = self.dict_to_device(data)
            batch_size = data[ChannelEnum.OCC_DEM].size(0)
            grid_size = list(data[ChannelEnum.OCC_DEM].size()[1:3])

            grid_data = data
            if subgrid_size is not None:
                data = self.split_subgrids(subgrid_size, data)

            if isinstance(self.model, LsqPlaneFitBaseline):
                # the profiler somehow has issues with the scipy lsq solver
                output = self.model.forward_pass(data)
            else:
                with profiler.record_function("model_inference"):
                    output = self.model.forward_pass(data)

            if subgrid_size is not None:
                # max occlusion ratio threshold for COMP_DEM where we accept reconstruction
                # instead of just taking all OCC_DEM
                subgrid_max_occ_ratio_thresh = self.task.config.get("subgrid_max_occ_ratio_thresh", 1.0)
                if subgrid_max_occ_ratio_thresh < 1.0:
                    occ_dem = data[ChannelEnum.OCC_DEM]
                    occ_ratio = torch.isnan(occ_dem).sum(dim=(1, 2)) / (occ_dem.size(1) * occ_dem.size(2))
                    occ_ratio_selector = occ_ratio > subgrid_max_occ_ratio_thresh

                    comp_dem = output[ChannelEnum.COMP_DEM]
                    comp_dem[occ_ratio_selector, :, :] = occ_dem[occ_ratio_selector, :, :]
                    output[ChannelEnum.COMP_DEM] = comp_dem

                    if ChannelEnum.OCC_DATA_UM in data and ChannelEnum.COMP_DATA_UM in output:
                        occ_data_um = output[ChannelEnum.OCC_DATA_UM]
                        comp_data_um = output[ChannelEnum.COMP_DATA_UM]
                        comp_data_um[occ_ratio_selector, :, :] = occ_data_um[occ_ratio_selector, :, :]
                        output[ChannelEnum.COMP_DATA_UM] = comp_dem

                output = self.unsplit_subgrids(grid_size, output)
                data = grid_data

            self.add_batch_data_to_hdf5_results(data_hdf5_group, data, start_idx, dataloader_meta_info.length)
            self.add_batch_data_to_hdf5_results(data_hdf5_group, output, start_idx, dataloader_meta_info.length)

            start_idx += batch_size
            progress_bar.next()
        progress_bar.finish()

        if not isinstance(self.model, LsqPlaneFitBaseline):
            prof.__exit__(0, None, None)
            with open(str(self.task.logdir / "inference_cputime.txt"), "a") as f:
                f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            if self.task.config.get("profiler_export_chrome_trace", False):
                prof.export_chrome_trace(str(self.task.logdir / "inference_cputime_chrome_trace.json"))

    def dict_to_device(self, data: Dict[Union[ChannelEnum, str], torch.Tensor]) \
            -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        for key, value in data.items():
            data[key] = value.to(self.device)
        return data

    def trace_model(self):
        assert self.task is not None

        self.logger.info(f"Tracing model for task {self.task.uid}")
        # TODO: this function is not adapted to model & data uncertainty estimation
        self.model.eval()

        if self.task.type in [TaskTypeEnum.SUPERVISED_LEARNING, TaskTypeEnum.INFERENCE]:
            dataloader = self.task.labeled_dataloader.dataloaders['test']
        else:
            raise NotImplementedError(f"The following task type is not implemented: {self.task.type}")

        for batch_idx, data in enumerate(dataloader):
            data = self.dict_to_device(data)
            batch_size = data[ChannelEnum.GT_DEM].size(0)

            input, norm_consts = self.model.assemble_input(data)

            traced = torch.jit.trace(self.model, input)

            # we only need to trace one batch
            break

        traced.save(str(self.task.logdir / "traced_model.pt"))

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

    @staticmethod
    def split_subgrids(subgrid_size: list, data: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        assert type(subgrid_size) == list and len(subgrid_size) == 2

        subgrid_data = {}
        for channel, tensor in data.items():
            # we assert batch size of 1 for now to make it easier
            assert tensor.size(0) == 1

            if channel in [ChannelEnum.OCC_DEM, ChannelEnum.GT_DEM, ChannelEnum.OCC_MASK, ChannelEnum.OCC_DATA_UM]:
                tensor = tensor.squeeze(dim=0)

                split_tensors_x = torch.split(tensor, subgrid_size[0], dim=0)

                subgrids = []
                for split_tensor_x in split_tensors_x:
                    subgrids += torch.split(split_tensor_x, subgrid_size[1], dim=1)

                tensor = torch.stack(subgrids)

            subgrid_data[channel] = tensor

        return subgrid_data

    @staticmethod
    def unsplit_subgrids(grid_size: list,
                         subgrid_output: Dict[ChannelEnum, torch.Tensor]) -> Dict[ChannelEnum, torch.Tensor]:
        output = {}
        for channel, tensor in subgrid_output.items():
            if channel in [ChannelEnum.REC_DEM, ChannelEnum.COMP_DEM, ChannelEnum.REC_DATA_UM, ChannelEnum.COMP_DATA_UM,
                           ChannelEnum.TOTAL_UM, ChannelEnum.REC_DEMS, ChannelEnum.COMP_DEMS,
                           ChannelEnum.REC_TRAV_RISK_MAP, ChannelEnum.COMP_TRAV_RISK_MAP]:
                grid = tensor.new_zeros(size=(1, grid_size[0], grid_size[1]))
                start_u = 0
                start_v = 0
                for subgrid_idx in range(tensor.size(0)):
                    subgrid = tensor[subgrid_idx]

                    stop_u = start_u + subgrid.size(0)
                    stop_v = start_v + subgrid.size(1)

                    grid[0, start_u:stop_u, start_v:stop_v] = subgrid

                    if stop_v >= grid.size(1):
                        # we jump to next major row
                        start_u = stop_u
                        start_v = 0
                    else:
                        start_v = stop_v

                tensor = grid

            output[channel] = tensor

        return output
