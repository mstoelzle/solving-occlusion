from abc import ABC, abstractmethod
from distutils.version import StrictVersion
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import *

from src.dataloaders.dataloader_meta_info import DataloaderMetaInfo
from src.enums import *
from src.learning.models.adf import adf
from src.learning.loss.loss import masked_loss_fct, mse_loss_fct, \
    l1_loss_fct, psnr_loss_fct, ssim_loss_fct, perceptual_loss_fct, style_loss_fct, log_likelihood_fct
from src.learning.normalization.input_normalization import InputNormalization


class BaseModel(ABC, nn.Module):
    def __init__(self, strict_forward_def: bool = True, **kwargs):
        super().__init__()
        self.config = kwargs

        self.input_dim: List = self.config["input_dim"]

        # models with strict forward definitions only take torch.Tensors as input
        self.strict_forward_def = strict_forward_def

    def train(self, mode: bool = True):
        """
        Attention: This method is quite resource-intensive. Only call if needed
        """
        super().train(mode=mode)

        self.dropout_mode = mode

    @abstractmethod
    def forward_pass(self, data: Dict[Union[ChannelEnum, str], torch.Tensor],
                     **kwargs) -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        pass

    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass

    def create_composed_map(self, occ_dem: torch.Tensor,
                            rec_dem: torch.Tensor) -> torch.Tensor:
        comp_dem = occ_dem.clone()

        selector = torch.isnan(occ_dem)
        comp_dem[selector] = rec_dem[selector]

        return comp_dem

    def get_normalized_data(self,
                            loss_config: dict,
                            output: Dict[Union[ChannelEnum, str], torch.Tensor],
                            data: Dict[ChannelEnum, torch.Tensor],
                            **kwargs) -> Dict[ChannelEnum, torch.Tensor]:
        normalization_config = loss_config.get("normalization", [])
        norm_data = {}
        if LossEnum.MSE_REC_ALL.value in normalization_config or \
                LossEnum.MSE_REC_OCC.value in normalization_config or \
                LossEnum.MSE_REC_NOCC.value in normalization_config:
            gt_dem = data[ChannelEnum.GT_DEM]
            rec_dem = output[ChannelEnum.REC_DEM]
            norm_elevation_map, ground_truth_norm_consts = InputNormalization.normalize(
                ChannelEnum.GT_DEM,
                input=gt_dem,
                batch=True,
                mean=True, stdev=True)
            norm_rec_dem, _ = InputNormalization.normalize(ChannelEnum.REC_DEM,
                                                           input=rec_dem,
                                                           batch=True, mean=True, stdev=True,
                                                           norm_consts=ground_truth_norm_consts)
            norm_data[ChannelEnum.GT_DEM] = norm_elevation_map
            norm_data[ChannelEnum.REC_DEM] = norm_rec_dem

        return norm_data

    def eval_loss_function(self,
                           loss_config: dict,
                           output: Dict[Union[ChannelEnum, str], torch.Tensor],
                           data: Dict[ChannelEnum, torch.Tensor],
                           dataloader_meta_info: DataloaderMetaInfo = None,
                           **kwargs) -> dict:
        sample_tensor = output[ChannelEnum.REC_DEM]

        loss_dict = {}

        if ChannelEnum.GT_DEM not in data:
            return {LossEnum.LOSS: sample_tensor.new_tensor(0),
                    LossEnum.L1_REC_ALL: sample_tensor.new_tensor(0),
                    LossEnum.L1_REC_OCC: sample_tensor.new_tensor(0),
                    LossEnum.L1_REC_NOCC: sample_tensor.new_tensor(0),
                    LossEnum.L1_COMP_ALL: sample_tensor.new_tensor(0),
                    LossEnum.MSE_REC_ALL: sample_tensor.new_tensor(0),
                    LossEnum.MSE_REC_OCC: sample_tensor.new_tensor(0),
                    LossEnum.MSE_REC_NOCC: sample_tensor.new_tensor(0),
                    LossEnum.MSE_COMP_ALL: sample_tensor.new_tensor(0),
                    LossEnum.SSIM_REC: sample_tensor.new_tensor(0),
                    LossEnum.SSIM_COMP: sample_tensor.new_tensor(0)}

        norm_data = self.get_normalized_data(loss_config, output, data, **kwargs)

        if LossEnum.L1_REC_ALL in norm_data:
            loss_dict[LossEnum.L1_REC_ALL] = l1_loss_fct(norm_data[ChannelEnum.REC_DEM],
                                                         norm_data[ChannelEnum.GT_DEM], **kwargs)
        else:
            loss_dict[LossEnum.L1_REC_ALL] = l1_loss_fct(output[ChannelEnum.REC_DEM],
                                                         data[ChannelEnum.GT_DEM], **kwargs)

        if LossEnum.L1_REC_OCC in norm_data:
            loss_dict[LossEnum.L1_REC_OCC] = masked_loss_fct(
                l1_loss_fct,
                norm_data[ChannelEnum.REC_DEM],
                norm_data[ChannelEnum.GT_DEM],
                data[ChannelEnum.OCC_MASK],
                **kwargs)
        else:
            loss_dict[LossEnum.L1_REC_OCC] = masked_loss_fct(
                l1_loss_fct,
                output[ChannelEnum.REC_DEM],
                data[ChannelEnum.GT_DEM],
                data[ChannelEnum.OCC_MASK],
                **kwargs)

        if LossEnum.L1_REC_NOCC in norm_data:
            loss_dict[LossEnum.L1_REC_NOCC] = masked_loss_fct(
                l1_loss_fct,
                norm_data[ChannelEnum.REC_DEM],
                norm_data[ChannelEnum.GT_DEM],
                ~data[ChannelEnum.OCC_MASK],
                **kwargs)
        else:
            loss_dict[LossEnum.L1_REC_NOCC] = masked_loss_fct(
                l1_loss_fct,
                output[ChannelEnum.REC_DEM],
                data[ChannelEnum.GT_DEM],
                ~data[ChannelEnum.OCC_MASK],
                **kwargs)

        if LossEnum.L1_COMP_ALL in norm_data:
            loss_dict[LossEnum.L1_COMP_ALL] = l1_loss_fct(norm_data[ChannelEnum.COMP_DEM],
                                                          norm_data[ChannelEnum.GT_DEM], **kwargs)
        else:
            loss_dict[LossEnum.L1_COMP_ALL] = l1_loss_fct(output[ChannelEnum.COMP_DEM],
                                                          data[ChannelEnum.GT_DEM], **kwargs)

        if LossEnum.MSE_REC_ALL in norm_data:
            loss_dict[LossEnum.MSE_REC_ALL] = mse_loss_fct(norm_data[ChannelEnum.REC_DEM],
                                                           norm_data[ChannelEnum.GT_DEM], **kwargs)
        else:
            loss_dict[LossEnum.MSE_REC_ALL] = mse_loss_fct(output[ChannelEnum.REC_DEM],
                                                           data[ChannelEnum.GT_DEM], **kwargs)

        if LossEnum.MSE_REC_OCC in norm_data:
            loss_dict[LossEnum.MSE_REC_OCC] = masked_loss_fct(
                mse_loss_fct,
                norm_data[ChannelEnum.REC_DEM],
                norm_data[ChannelEnum.GT_DEM],
                data[ChannelEnum.OCC_MASK],
                **kwargs)
        else:
            loss_dict[LossEnum.MSE_REC_OCC] = masked_loss_fct(
                mse_loss_fct,
                output[ChannelEnum.REC_DEM],
                data[ChannelEnum.GT_DEM],
                data[ChannelEnum.OCC_MASK],
                **kwargs)

        if LossEnum.MSE_REC_NOCC in norm_data:
            loss_dict[LossEnum.MSE_REC_NOCC] = masked_loss_fct(
                mse_loss_fct,
                norm_data[ChannelEnum.REC_DEM],
                norm_data[ChannelEnum.GT_DEM],
                ~data[ChannelEnum.OCC_MASK],
                **kwargs)
        else:
            loss_dict[LossEnum.MSE_REC_NOCC] = masked_loss_fct(
                mse_loss_fct,
                output[ChannelEnum.REC_DEM],
                data[ChannelEnum.GT_DEM],
                ~data[ChannelEnum.OCC_MASK],
                **kwargs)

        if LossEnum.MSE_COMP_ALL in norm_data:
            loss_dict[LossEnum.MSE_COMP_ALL] = mse_loss_fct(norm_data[ChannelEnum.COMP_DEM],
                                                            norm_data[ChannelEnum.GT_DEM], **kwargs)
        else:
            loss_dict[LossEnum.MSE_COMP_ALL] = mse_loss_fct(output[ChannelEnum.COMP_DEM],
                                                            data[ChannelEnum.GT_DEM], **kwargs)

        if LossEnum.SSIM_REC in norm_data:
            loss_dict[LossEnum.SSIM_REC] = ssim_loss_fct(norm_data[ChannelEnum.REC_DEM],
                                                         norm_data[ChannelEnum.GT_DEM],
                                                         data_min=dataloader_meta_info.min,
                                                         data_max=dataloader_meta_info.max,
                                                         **kwargs)
        else:
            loss_dict[LossEnum.SSIM_REC] = ssim_loss_fct(output[ChannelEnum.REC_DEM],
                                                         data[ChannelEnum.GT_DEM],
                                                         data_min=dataloader_meta_info.min,
                                                         data_max=dataloader_meta_info.max,
                                                         **kwargs)
        if LossEnum.SSIM_COMP in norm_data:
            loss_dict[LossEnum.SSIM_COMP] = ssim_loss_fct(norm_data[ChannelEnum.COMP_DEM],
                                                          norm_data[ChannelEnum.GT_DEM],
                                                          data_min=dataloader_meta_info.min,
                                                          data_max=dataloader_meta_info.max,
                                                          **kwargs)
        else:
            loss_dict[LossEnum.SSIM_COMP] = ssim_loss_fct(output[ChannelEnum.COMP_DEM],
                                                          data[ChannelEnum.GT_DEM],
                                                          data_min=dataloader_meta_info.min,
                                                          data_max=dataloader_meta_info.max,
                                                          **kwargs)

        if ChannelEnum.REC_DATA_UM in output:
            nll_rec_data = -log_likelihood_fct(input_mean=output[ChannelEnum.REC_DEM],
                                               input_variance=output[ChannelEnum.REC_DATA_UM],
                                               target=data[ChannelEnum.GT_DEM], **kwargs)
            loss_dict[LossEnum.NLL_REC_DATA] = nll_rec_data

            # MSE error between input, occluded data uncertainty and reconstructed data uncertainty
            # masked for non-occluded area
            mse_rec_data_um_nocc = loss_dict[LossEnum.MSE_REC_NOCC] = masked_loss_fct(
                mse_loss_fct,
                output[ChannelEnum.REC_DATA_UM],
                data[ChannelEnum.OCC_DATA_UM],
                ~data[ChannelEnum.OCC_MASK],
                **kwargs)
            loss_dict[LossEnum.MSE_REC_DATA_UM_NOCC] = mse_rec_data_um_nocc

        if ChannelEnum.COMP_DATA_UM in output:
            nll_comp_data = -log_likelihood_fct(input_mean=output[ChannelEnum.COMP_DEM],
                                                input_variance=output[ChannelEnum.COMP_DATA_UM],
                                                target=data[ChannelEnum.GT_DEM], **kwargs)
            loss_dict[LossEnum.NLL_COMP_DATA] = nll_comp_data
        if ChannelEnum.MODEL_UM in output:
            nll_model = -log_likelihood_fct(input_mean=output[ChannelEnum.REC_DEM],
                                            input_variance=output[ChannelEnum.MODEL_UM],
                                            target=data[ChannelEnum.GT_DEM], **kwargs)
            loss_dict[LossEnum.NLL_MODEL] = nll_model
        if ChannelEnum.TOTAL_UM in output:
            nll_total = -log_likelihood_fct(input_mean=output[ChannelEnum.REC_DEM],
                                            input_variance=output[ChannelEnum.TOTAL_UM],
                                            target=data[ChannelEnum.GT_DEM], **kwargs)
            loss_dict[LossEnum.NLL_TOTAL] = nll_total

        weights = loss_config.get("eval_weights", {})
        recons_weight = weights.get(LossEnum.MSE_REC_ALL.value, 0)
        recons_occlusion_weight = weights.get(LossEnum.MSE_REC_OCC.value, 1)
        recons_non_occlusion_weight = weights.get(LossEnum.MSE_REC_NOCC.value, 0)

        loss_dict[LossEnum.LOSS] = recons_weight * loss_dict[LossEnum.MSE_REC_ALL] + \
                                   recons_occlusion_weight * loss_dict[LossEnum.MSE_REC_OCC] + \
                                   recons_non_occlusion_weight * loss_dict[LossEnum.MSE_REC_NOCC]

        return loss_dict

    def artistic_loss_function(self,
                               feature_extractor: nn.Module,
                               loss_config: dict,
                               output: Dict[Union[ChannelEnum, str], torch.Tensor],
                               data: Dict[ChannelEnum, torch.Tensor],
                               **kwargs) -> dict:
        # Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge.
        # "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).
        # https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/loss.py
        # this requires a feature_extractor

        perceptual_loss = 0.
        style_loss = 0.
        if feature_extractor is not None:
            gt_dem = data[ChannelEnum.GT_DEM]
            rec_dem = output[ChannelEnum.REC_DEM]
            comp_dem = output[ChannelEnum.COMP_DEM]

            # features for the ground-truth, the reconstructed elevation map and the inpainted elevation map
            # the feature extractor expects an image with three channels as an input
            feat_gt = feature_extractor(gt_dem.unsqueeze(dim=1).repeat(1, 3, 1, 1))
            feat_rec = feature_extractor(rec_dem.unsqueeze(dim=1).repeat(1, 3, 1, 1))
            feat_comp = feature_extractor(comp_dem.unsqueeze(dim=1).repeat(1, 3, 1, 1))

            for i in range(len(feat_gt)):
                perceptual_loss += perceptual_loss_fct(feat_rec[i], feat_gt[i], **kwargs)
                perceptual_loss += perceptual_loss_fct(feat_comp[i], feat_gt[i], **kwargs)

                style_loss += style_loss_fct(feat_rec[i], feat_gt[i], **kwargs)
                style_loss += style_loss_fct(feat_comp[i], feat_gt[i], **kwargs)

        return {LossEnum.PERCEPTUAL: perceptual_loss,
                LossEnum.STYLE: style_loss}

    def set_dropout_mode(self, dropout_mode: bool = False):
        if self.dropout_mode != dropout_mode:
            self.apply(lambda m: apply_dropout(m, dropout_mode))
            self.dropout_mode = dropout_mode


def apply_dropout(m: nn.Module, dropout_mode: bool = False):
    if m.__class__.__name__.startswith('Dropout'):
        if dropout_mode:
            m.train()
        else:
            m.eval()
