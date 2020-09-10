import csv
import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F

from .domain_distance_metrics.cmmd import conditional_maximum_mean_discrepancy
from .domain_distance_metrics.coral import coral
from .domain_distance_metrics.fid import frechet_inception_distance
from .domain_distance_metrics.lmmd import local_maximum_mean_discrepancy
from .domain_distance_metrics.mmd import maximum_mean_discrepancy
from src.enums import *
from src.utils.log import get_logger

logger = get_logger("loss")


class Loss(ABC):
    def __init__(self, logdir: pathlib.Path, **kwargs):
        self.config = kwargs
        self.logdir = logdir

        self.report_frequency: int = self.config["report_frequency"]
        self.batch_results = []  # must be reset at the beginning of each epoch
        self.batch_sizes = []

        self.epoch_losses = {"train": {}, "val": {}, "test": {}}

        self.purpose: Optional[str] = None
        self.epoch: Optional[int] = None

    def new_epoch(self, epoch: int, purpose: str):
        self.purpose = purpose
        self.epoch = epoch
        return self

    def __enter__(self):
        self.batch_results = []
        self.batch_sizes = []

    def __call__(self, batch_size: int, loss_dict: dict):
        # assert that a loss (total loss) key is available
        assert LossEnum.LOSS in loss_dict

        # aggregate results
        self.batch_results.append(loss_dict)
        self.batch_sizes.append(batch_size)

        if len(self.batch_results) % self.report_frequency == 0 and self.purpose == "train":
            logger.info(f"Epoch {self.epoch} - Batch {len(self.batch_results)} - {self.purpose} loss "
                        f"- {loss_dict[LossEnum.LOSS]:.4f}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        epoch_result = self.compute_epoch_result()
        self.epoch_losses[self.purpose].update({str(self.epoch): epoch_result[LossEnum.LOSS]})

        if self.purpose != "test":
            logger.info(f"Epoch {self.epoch} - {self.purpose} average loss - {epoch_result[LossEnum.LOSS]:.4f}")
        else:
            logger.info(f"Testing Loss: {epoch_result[LossEnum.LOSS]:.4f}")

        self.write_logfile(self.epoch, epoch_result)

    def write_logfile(self, epoch, epoch_result: Dict):
        logfile_path = self.logdir / f"{self.purpose}_losses.csv"
        if logfile_path.exists():
            write_mode = "a"
        else:
            write_mode = "w"

        write_dict = {"epoch": epoch}
        for key, value in epoch_result.items():
            if type(value) == torch.Tensor:
                value = value.item()

            if type(key) == LossEnum:
                write_dict[key.value] = value
            else:
                write_dict[key] = value

        with open(str(logfile_path), write_mode, newline='') as fp:
            fieldnames = list(write_dict.keys())
            writer = csv.DictWriter(fp, fieldnames=fieldnames)

            if write_mode == "w":
                writer.writeheader()

            writer.writerow(write_dict)

    def get_epoch_loss(self):
        return self.epoch_losses[self.purpose][str(self.epoch)]

    def compute_epoch_result(self) -> Dict:
        num_samples: int = 0
        total_loss_dict = None

        for batch_idx, (batch_size, batch_result) in enumerate(zip(self.batch_sizes, self.batch_results)):
            if total_loss_dict is None:
                total_loss_dict = {}
                for key, loss in batch_result.items():
                    total_loss_dict[key] = batch_result[key] * batch_size
            else:
                # assert that all loss dicts have the same structure
                assert total_loss_dict.keys() == batch_result.keys()

                for key, loss in batch_result.items():
                    total_loss_dict[key] += batch_result[key] * batch_size

            num_samples += batch_size

        epoch_loss_dict = {}
        for key, loss in total_loss_dict.items():
            epoch_loss_dict[key] = total_loss_dict[key] / float(num_samples)

        return epoch_loss_dict


def reconstruction_occlusion_loss_fct(reconstructed_elevation_map: torch.Tensor,
                                      elevation_map: torch.Tensor,
                                      binary_occlusion_map: torch.Tensor):

    recons_loss = F.mse_loss(reconstructed_elevation_map[binary_occlusion_map == 1],
                             elevation_map[binary_occlusion_map == 1])
    return recons_loss


def kld_loss_fct(mu: torch.Tensor, log_var: torch.Tensor):
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    return kld_loss


def calc_domain_distance(data_domain_1: torch.Tensor, data_domain_2: torch.Tensor,
                         domain_distance_metric: str = "fid", normalize_activations=True, num_classes=None,
                         targets_domain_1: torch.Tensor = None, targets_domain_2: torch.Tensor = None,
                         pred_targets_domain_1: torch.Tensor = None,
                         pred_targets_domain_2: torch.Tensor = None) -> torch.Tensor:
    # Input: two 2D torch tensors (e.g. vectors) with the dimension (batch_size, feature_size)
    # In the case of lmmd, the order of the label predictions is important:
    # Per definition, the labeled source domain is defined as domain 1 and the unlabeled target domain as domain 2
    batch_size = data_domain_1.size(0)
    layer_dim = data_domain_1.size(1)

    assert data_domain_1.size() == data_domain_2.size(), "The tensors from both domains need to have the same size."

    if torch.isnan(data_domain_1).any() or torch.isnan(data_domain_2).any():
        logger.error("The tensor_domain_1 and / or tensor_domain_2 contains NaN elements in calc_domain_distance\n"
                     "--> skipping the domain distance")
        return data_domain_1.new_tensor(0.0)

    # weather to normalize the activations by the own magnitude
    if normalize_activations:
        # activation average over all batches from both domains
        mean_activations = torch.mean(torch.cat((data_domain_1, data_domain_2), dim=0), dim=0)
        # sum together over all activations
        sum_mean_activations = torch.sum(torch.abs(mean_activations))

        data_domain_1_normalized = data_domain_1 / sum_mean_activations
        data_domain_2_normalized = data_domain_2 / sum_mean_activations
    else:
        data_domain_1_normalized = data_domain_1 / layer_dim
        data_domain_2_normalized = data_domain_2 / layer_dim

    if domain_distance_metric == "fid":
        domain_distance = frechet_inception_distance(data_domain_1_normalized, data_domain_2_normalized)

        if normalize_activations:
            domain_distance = 10. ** 4 * domain_distance
        else:
            domain_distance = 10. ** 2 * domain_distance

    elif domain_distance_metric == "fid_legacy":
        domain_distance = frechet_inception_distance(data_domain_1, data_domain_2)

    elif domain_distance_metric == "mean_diff":
        mean_domain_1 = torch.mean(data_domain_1_normalized, dim=0)
        mean_domain_2 = torch.mean(data_domain_2_normalized, dim=0)

        # we are calculating the p=2 norm of the mean activations
        domain_distance = torch.dist(mean_domain_1, mean_domain_2, p=2)

    elif domain_distance_metric == "mengfan_wu":
        # Mengfan Wu's implementation
        std_labeled, mean_labeled = torch.std_mean(data_domain_1_normalized, dim=0)
        std_unlabeled, mean_unlabeled = torch.std_mean(data_domain_2_normalized, dim=0)
        diff_mean = torch.dist(mean_labeled, mean_unlabeled)
        diff_std = torch.dist(std_labeled, std_unlabeled, p=1)

        domain_distance = 10. * diff_mean + diff_std

    elif domain_distance_metric == "mmd":
        # the alternative option is to use the torch-two-sample library
        # mmd_statistic = MMDStatistic(batch_size, batch_size)
        # domain_distance = mmd_statistic(domain_1_normalized, domain_2_normalized, alphas=[1. / 2.])**2

        domain_distance = maximum_mean_discrepancy(data_domain_1, data_domain_2) ** 2

    elif domain_distance_metric == "cosine_similarity":
        mean_domain_1 = torch.mean(data_domain_1_normalized, dim=0)
        mean_domain_2 = torch.mean(data_domain_2_normalized, dim=0)

        domain_distance = data_domain_1.new_tensor(1.0) - F.cosine_similarity(mean_domain_1, mean_domain_2, dim=0)

    elif domain_distance_metric == "coral":
        domain_distance = coral(data_domain_1, data_domain_2)

    elif domain_distance_metric == "cmmd":
        domain_distance = conditional_maximum_mean_discrepancy(num_classes,
                                                               data_domain_1, data_domain_2,
                                                               targets_domain_1, targets_domain_2)

    elif domain_distance_metric == "lmmd":
        # In the case of lmmd, the order of the label predictions is important:
        # Per definition, the labeled source domain is defined as domain 1 and the unlabeled target domain as domain 2
        domain_distance = local_maximum_mean_discrepancy(num_classes,
                                                         data_domain_1, data_domain_2,
                                                         targets_domain_1, pred_targets_domain_2)

    else:
        raise NotImplementedError(f"The domain distance metric {domain_distance_metric} is not implemented.")

    return domain_distance
