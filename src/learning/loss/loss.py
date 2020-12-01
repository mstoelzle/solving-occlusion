from abc import ABC, abstractmethod
import csv
import logging
import numpy as np
import pathlib
from pytorch_msssim import ssim
from typing import Callable, Dict, List, Optional, Tuple
import warnings

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import torchvision

from .domain_distance_metrics.cmmd import conditional_maximum_mean_discrepancy
from .domain_distance_metrics.coral import coral
from .domain_distance_metrics.fid import frechet_inception_distance
from .domain_distance_metrics.lmmd import local_maximum_mean_discrepancy
from .domain_distance_metrics.mmd import maximum_mean_discrepancy
from src.dataloaders.dataloader_meta_info import DataloaderMetaInfo
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
        self.dataloader_meta_info: Optional[DataloaderMetaInfo] = None

    def new_epoch(self, epoch: int, purpose: str, dataloader_meta_info: DataloaderMetaInfo = None):
        self.purpose = purpose
        self.epoch = epoch
        self.dataloader_meta_info = dataloader_meta_info

        return self

    def __enter__(self):
        self.batch_results = []
        self.batch_sizes = []

    def __call__(self, batch_size: int, loss_dict: dict):
        # assert that a loss (total loss) key is available
        assert LossEnum.LOSS in loss_dict

        loss_dict_cuda = loss_dict
        loss_dict = {}
        for key, value in loss_dict_cuda.items():
            if type(value) == torch.Tensor:
                loss_dict[key] = value.detach().cpu()

        # aggregate results
        self.batch_results.append(loss_dict)
        self.batch_sizes.append(batch_size)

        # if len(self.batch_results) % self.report_frequency == 0 and self.purpose == "train":
        #     logger.info(f"Epoch {self.epoch} - Batch {len(self.batch_results)} - {self.purpose} loss "
        #                 f"- {loss_dict[LossEnum.LOSS]:.4f}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        epoch_result = self.compute_epoch_result()
        self.epoch_losses[self.purpose].update({str(self.epoch): epoch_result[LossEnum.LOSS]})

        if self.purpose != "test":
            logger.info(f"Epoch {self.epoch} - {self.purpose} average loss - {epoch_result[LossEnum.LOSS]:.4f}")
        else:
            logger.info(f"Testing Loss: {epoch_result[LossEnum.LOSS]:.4f}")

        self.write_logfile(self.epoch, epoch_result)

        self.dataloader_meta_info = None

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

        if self.dataloader_meta_info is not None:
            mse_psnr_enum_mapping = {
                LossEnum.MSE_REC_ALL: LossEnum.PSNR_REC_ALL,
                LossEnum.MSE_REC_OCC: LossEnum.PSNR_REC_OCC,
                LossEnum.MSE_REC_NOCC: LossEnum.PSNR_REC_NOCC,
                LossEnum.MSE_COMP_ALL: LossEnum.PSNR_COMP_ALL
            }

            for mse_loss_enum in mse_psnr_enum_mapping.keys():
                psnr_loss_enum = mse_psnr_enum_mapping[mse_loss_enum]

                mse_loss = epoch_loss_dict[mse_loss_enum]
                psnr_loss = psnr_from_mse_loss_fct(mse_loss, self.dataloader_meta_info.min,
                                                   self.dataloader_meta_info.max)

                epoch_loss_dict[psnr_loss_enum] = psnr_loss.mean()

        return epoch_loss_dict

    def aggregate_mean_loss_dict(self, loss_dict: dict) -> dict:
        aggregated_loss_dict = {}
        for key, value in loss_dict.items():
            if type(value) == torch.Tensor:
                aggregated_loss_dict[key] = value.mean()

        return aggregated_loss_dict


def reduction_fct(loss: torch.Tensor, reduction='mean', **kwargs) -> torch.Tensor:
    if reduction == 'mean_per_sample':
        if loss.dim() > 1:
            loss = loss.mean(dim=tuple(range(1, loss.dim())))
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError

    return loss


def masked_loss_fct(loss_fct: Callable, input: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor, **kwargs):
    loss = loss_fct(input, target, reduction="none")

    selector = (~torch.isnan(input)) & (~torch.isnan(target)) & (mask == 1)
    loss[~selector] = 0

    loss = reduction_fct(loss, **kwargs)

    return loss


def l1_loss_fct(input: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    l1_loss = F.l1_loss(input, target, reduction="none")

    selector = (~torch.isnan(input)) & (~torch.isnan(target))
    l1_loss[~selector] = 0

    l1_loss = reduction_fct(l1_loss, **kwargs)

    return l1_loss


def mse_loss_fct(input: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    mse_loss = F.mse_loss(input, target, reduction="none")

    selector = (~torch.isnan(input)) & (~torch.isnan(target))
    mse_loss[~selector] = 0

    mse_loss = reduction_fct(mse_loss, **kwargs)

    return mse_loss


# peak signal-to-noise ratio
def psnr_loss_fct(input, target, data_min: float, data_max: float, **kwargs) -> torch.Tensor:
    # normalize for minimum being at 0
    input_off = input - data_min
    target_off = target - data_min

    mse = mse_loss_fct(input_off, target_off, **kwargs)

    return psnr_from_mse_loss_fct(mse, data_min, data_max, **kwargs)


def psnr_from_mse_loss_fct(mse: torch.Tensor, data_min: float, data_max: float, **kwargs):
    dynamic_range = data_max - data_min

    # handle divisions by zero
    # the PSNR is infinity for MSE equal to zero
    psnr = mse.new_ones(size=mse.size()) * np.Inf
    selector = (mse != 0)

    psnr[selector] = 20 * torch.log10(dynamic_range / torch.sqrt(mse[selector]))

    return psnr


# structural similarity index (SSIM)
# https://github.com/VainF/pytorch-msssim
# https://en.wikipedia.org/wiki/Structural_similarity
def ssim_loss_fct(input, target, data_min: float, data_max: float, **kwargs) -> torch.Tensor:
    dynamic_range = data_max - data_min

    # normalize for minimum being at 0
    input_off = input - data_min
    target_off = target - data_min

    if torch.isnan(input).sum() > 0 or torch.isnan(target).sum() > 0:
        warnings.warn("The SSIM is not reliable when called upon tensors with nan values")
        ssim_loss = input.new_zeros(size=(input.size(0), ))
    else:
        ssim_loss = ssim(input_off.unsqueeze(1), target_off.unsqueeze(1),
                         data_range=dynamic_range, size_average=False)

    ssim_loss = reduction_fct(ssim_loss, **kwargs)

    return ssim_loss


def kld_loss_fct(mu: torch.Tensor, log_var: torch.Tensor):
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    return kld_loss


def total_variation_loss_fct(image: torch.Tensor):
    """
    Total variation loss of an image
    Source:
    Johnson, Justin, Alexandre Alahi, and Li Fei-Fei.
    "Perceptual losses for real-time style transfer and super-resolution."
    European conference on computer vision. Springer, Cham, 2016.
    https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/loss.py
    :param image:
    :return:
    """
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[..., :, :-1] - image[..., :, 1:])) + \
           torch.mean(torch.abs(image[..., :-1, :] - image[..., 1:, :]))
    return loss


def masked_total_variation_loss_fct(input: torch.Tensor, mask: torch.Tensor):
    """
    Total variation loss of a masked image
    Source:
    Johnson, Justin, Alexandre Alahi, and Li Fei-Fei.
    "Perceptual losses for real-time style transfer and super-resolution."
    European conference on computer vision. Springer, Cham, 2016.
    Adapted from:
    https://github.com/ryanwongsa/Image-Inpainting/blob/master/src/loss/loss_compute.py
    :param mask:
    :param input:
    :return:
    """

    # we need to ensure that we also include NaN values in the input into the mask
    mask = mask | torch.isnan(input)

    if input.dim() == 3:
        # we need a channel composition foe 2d convolutions
        input = input.unsqueeze(dim=1)

    if mask.dim() == 3:
        # we need a channel composition foe 2d convolutions
        mask = mask.unsqueeze(dim=1)

    if mask.dtype == torch.bool:
        mask = mask.to(dtype=torch.float)

    kernel = torch.ones((input.size(1), input.size(1), mask.shape[1], mask.shape[1]), requires_grad=False)
    kernel = kernel.to(device=input.device)
    dilated_mask = F.conv2d(mask, weight=kernel, padding=1) > 0
    dilated_mask = dilated_mask.clone().detach().float()

    P = dilated_mask[..., 1:-1, 1:-1] * input

    a = torch.mean(torch.abs(P[:, :, :, 1:] - P[:, :, :, :-1]))
    b = torch.mean(torch.abs(P[:, :, 1:, :] - P[:, :, :-1, :]))
    return a + b


def perceptual_loss_fct(input: torch.Tensor, target: torch.Tensor, **kwargs):
    return l1_loss_fct(input, target, **kwargs)


def style_loss_fct(input: torch.Tensor, target: torch.Tensor, **kwargs):
    return l1_loss_fct(gram_matrix(input), gram_matrix(target), **kwargs)


def gram_matrix(feat: torch.Tensor):
    """
    Auto correlation on feature map (Gram matrix)
    https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    :param feat:
    :return:
    """
    if feat.dim() == 3:
        # this code is made for the situation, where our feature tensor does not posses channels
        ch = 1
        (b, h, w) = feat.size()
    elif feat.dim() == 4:
        (b, ch, h, w) = feat.size()
    else:
        raise NotImplementedError

    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


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
