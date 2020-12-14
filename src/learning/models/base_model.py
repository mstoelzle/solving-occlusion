from abc import ABC, abstractmethod
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
    def __init__(self, seed: int, in_channels: List[str], out_channels: List[str],
                 input_normalization: Dict = None, **kwargs):
        super().__init__()
        self.config = kwargs
        self.seed = seed

        self.input_dim: List = self.config["input_dim"]

        self.in_channels = [ChannelEnum(in_channel) for in_channel in in_channels]
        self.out_channels = [ChannelEnum(out_channel) for out_channel in out_channels]

        self.input_normalization = None if input_normalization is False else input_normalization

        self.dropout_p = self.config.get("training_dropout_probability", 0.0)
        self.training_dropout = True if self.dropout_p > 0.0 else False

        self.adf = False
        self.keep_variance_fn = None
        if self.config.get("data_uncertainty_estimation") is not None:
            data_uncertainty_config = self.config["data_uncertainty_estimation"]
            self.data_uncertainty_method = DataUncertaintyMethodEnum(data_uncertainty_config["method"])
            if self.data_uncertainty_method == DataUncertaintyMethodEnum.ADF:
                self.adf = True
                min_variance = data_uncertainty_config.get("min_variance", 0.001)
                self.keep_variance_fn = lambda x: adf.keep_variance(x, min_variance=min_variance)
            else:
                raise NotImplementedError

        self.model_uncertainty_method = None
        self.num_solutions: int = 1
        if self.config.get("model_uncertainty_estimation") is not None:
            model_uncertainty_config = self.config["model_uncertainty_estimation"]
            self.model_uncertainty_method = ModelUncertaintyMethodEnum(model_uncertainty_config["method"])
            if self.model_uncertainty_method == ModelUncertaintyMethodEnum.MONTE_CARLO_DROPOUT:
                print("training dropout activated", self.training_dropout)
                if self.training_dropout:
                    assert self.dropout_p == model_uncertainty_config["probability"]
                else:
                    self.dropout_p = model_uncertainty_config["probability"]

                self.num_solutions = int(model_uncertainty_config["num_solutions"])
                self.use_mean_as_rec = model_uncertainty_config.get("use_mean_as_rec", False)
            elif self.model_uncertainty_method == ModelUncertaintyMethodEnum.MONTE_CARLO_VAE:
                self.num_solutions = int(model_uncertainty_config["num_solutions"])
                self.use_mean_as_rec = model_uncertainty_config.get("use_mean_as_rec", False)
            else:
                raise NotImplementedError

    def train(self, mode: bool = True, dropout_mode: bool = False):
        super().train(mode=mode)

        # PyTorch automatically deactivates dropout for evaluation
        # however for uncertainty estimation with monte carlo dropout we still need it during evaluation
        if mode != dropout_mode:
            for m in self.modules():
                # print(m.__class__.__name__)
                if m.__class__.__name__.startswith('Dropout'):
                    if dropout_mode:
                        m.train()
                    else:
                        m.eval()

    def forward(self, data: Dict[Union[ChannelEnum, str], torch.Tensor],
                **kwargs) -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        input, norm_consts = self.assemble_input(data)

        output = {}

        self.train(mode=self.training, dropout_mode=True if self.training and self.training_dropout else False)
        data_uncertainty = None
        x = self.forward_pass(input=input, data=data)
        if type(x) in [list, tuple]:
            rec_dem = x[0]
            data_uncertainty = x[1]
        else:
            rec_dem = x
        output[ChannelEnum.REC_DEM] = rec_dem

        model_uncertainty = None
        if self.num_solutions > 1 and self.training is False:
            if self.model_uncertainty_method == ModelUncertaintyMethodEnum.MONTE_CARLO_DROPOUT:
                self.train(mode=self.training, dropout_mode=True)

                dem_solutions = []
                data_uncertainties = []
                for i in range(self.num_solutions):
                    x = self.forward_pass(input=input, data=data)

                    if type(x) in [list, tuple]:
                        dem_solutions.append(x[0])
                        data_uncertainties.append(x[1])
                    else:
                        dem_solutions.append(x)

                dem_solutions = torch.stack(dem_solutions, dim=1)
                model_uncertainty = torch.var(dem_solutions, dim=1)

                if self.use_mean_as_rec:
                    output[ChannelEnum.REC_DEM] = torch.mean(dem_solutions, dim=1)

                    if len(data_uncertainties) > 0:
                        data_uncertainties = torch.stack(data_uncertainties, dim=1)
                        data_uncertainty = torch.mean(data_uncertainties, dim=1)
            else:
                raise NotImplementedError

            output[ChannelEnum.MODEL_UNCERTAINTY_MAP] = model_uncertainty
            output[ChannelEnum.REC_DEMS] = dem_solutions

        if data_uncertainty is not None:
            output[ChannelEnum.DATA_UNCERTAINTY_MAP] = data_uncertainty
        if model_uncertainty is not None:
            output[ChannelEnum.MODEL_UNCERTAINTY_MAP] = model_uncertainty

        if data_uncertainty is not None and model_uncertainty is not None:
            total_uncertainty = data_uncertainty + model_uncertainty
        elif data_uncertainty is not None:
            total_uncertainty = data_uncertainty
        elif model_uncertainty is not None:
            total_uncertainty = model_uncertainty
        else:
            total_uncertainty = None
        if total_uncertainty is not None:
            output[ChannelEnum.TOTAL_UNCERTAINTY_MAP] = total_uncertainty

        output = self.denormalize_output(data, output, norm_consts)

        return output

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass

    def assemble_input(self, data: Dict[Union[str, ChannelEnum], torch.Tensor]) \
            -> Tuple[Union[List, torch.Tensor], Dict]:
        input = None
        norm_consts = {}
        for channel_idx, in_channel in enumerate(self.in_channels):
            if in_channel in data:
                channel_data = data[in_channel]
            else:
                raise NotImplementedError

            if self.input_normalization is not None:
                if in_channel == ChannelEnum.OCC_DEM or \
                        in_channel == ChannelEnum.GT_DEM:
                    channel_data, norm_consts[in_channel] = InputNormalization.normalize(in_channel, channel_data,
                                                                                         **self.input_normalization,
                                                                                         batch=True)

            if in_channel == ChannelEnum.OCC_DEM:
                channel_data = self.preprocess_occluded_map(channel_data)

            if in_channel == ChannelEnum.OCC_MASK:
                channel_data = ~channel_data

            if input is None:
                input_size = (channel_data.size(0), len(self.in_channels), channel_data.size(1), channel_data.size(2))
                input = channel_data.new_zeros(size=input_size, dtype=torch.float32)

            input[:, channel_idx, ...] = channel_data.unsqueeze(dim=1)[:, 0, :, :]

        if self.adf:
            # TODO: implement actual data uncertainty estimation
            var = torch.ones_like(input) * 0.0001
            var[data[ChannelEnum.OCC_MASK] == 1] = np.NaN
            var = self.preprocess_occluded_map(var)
            input = [input, var]

        return input, norm_consts

    def preprocess_occluded_map(self, occ_dem: torch.Tensor) -> torch.Tensor:
        poem = occ_dem.clone()

        NaN_replacement = self.config.get("NaN_replacement", 0)

        # replace NaNs signifying occluded areas with arbitrary high or low number
        # poem[occ_dem != occ_dem] = -10000
        poem[occ_dem != occ_dem] = NaN_replacement

        # TODO: for torch==1.8
        # poem = torch.nan_to_num(occ_dem, nan=NaN_replacement)

        return poem

    def create_composed_elevation_map(self, occ_dem: torch.Tensor,
                                      rec_dem: torch.Tensor) -> torch.Tensor:
        comp_dem = occ_dem.clone()

        selector = torch.isnan(occ_dem)
        comp_dem[selector] = rec_dem[selector]

        return comp_dem

    def denormalize_output(self,
                           data: Dict[ChannelEnum, torch.Tensor],
                           output: Dict[Union[ChannelEnum, str], torch.Tensor],
                           norm_consts: dict) -> Dict[Union[ChannelEnum, str], torch.Tensor]:

        if self.input_normalization is not None:
            denorm_output = {}
            for key, value in output.items():
                if key in [ChannelEnum.REC_DEM, ChannelEnum.REC_DEMS]:
                    if ChannelEnum.GT_DEM in norm_consts:
                        denormalize_norm_const = norm_consts[ChannelEnum.GT_DEM]
                    elif ChannelEnum.OCC_DEM in norm_consts:
                        denormalize_norm_const = norm_consts[ChannelEnum.OCC_DEM]
                    else:
                        raise ValueError

                    denorm_output[key] = InputNormalization.denormalize(key, input=value, batch=True,
                                                                        norm_consts=denormalize_norm_const,
                                                                        **self.input_normalization)
                else:
                    denorm_output[key] = value
        else:
            denorm_output = output

        rec_dem = denorm_output[ChannelEnum.REC_DEM]
        comp_dem = self.create_composed_elevation_map(data[ChannelEnum.OCC_DEM], rec_dem)
        denorm_output[ChannelEnum.COMP_DEM] = comp_dem

        if ChannelEnum.REC_DEMS in denorm_output:
            rec_dems = denorm_output[ChannelEnum.REC_DEMS]
            occ_dems = []
            for i in range(rec_dems.size(dim=1)):
                occ_dems.append(data[ChannelEnum.OCC_DEM])
            occ_dems = torch.stack(occ_dems, dim=1)
            denorm_output[ChannelEnum.COMP_DEMS] = self.create_composed_elevation_map(occ_dems, rec_dems)

        return denorm_output

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

        if ChannelEnum.DATA_UNCERTAINTY_MAP in output:
            nll_data = -log_likelihood_fct(input_mean=output[ChannelEnum.REC_DEM],
                                           input_variance=output[ChannelEnum.DATA_UNCERTAINTY_MAP],
                                           target=data[ChannelEnum.GT_DEM], **kwargs)
            loss_dict[LossEnum.NLL_MODEL] = nll_data
        if ChannelEnum.MODEL_UNCERTAINTY_MAP in output:
            nll_model = -log_likelihood_fct(input_mean=output[ChannelEnum.REC_DEM],
                                            input_variance=output[ChannelEnum.MODEL_UNCERTAINTY_MAP],
                                            target=data[ChannelEnum.GT_DEM], **kwargs)
            loss_dict[LossEnum.NLL_MODEL] = nll_model
        if ChannelEnum.TOTAL_UNCERTAINTY_MAP in output:
            nll_total = -log_likelihood_fct(input_mean=output[ChannelEnum.REC_DEM],
                                            input_variance=output[ChannelEnum.TOTAL_UNCERTAINTY_MAP],
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
                               loss_config: dict,
                               output: Dict[Union[ChannelEnum, str], torch.Tensor],
                               data: Dict[ChannelEnum, torch.Tensor],
                               **kwargs) -> dict:
        # Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge.
        # "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).
        # https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/loss.py
        # this requires a self.feature_extractor

        perceptual_loss = 0.
        style_loss = 0.
        if self.feature_extractor is not None:
            gt_dem = data[ChannelEnum.GT_DEM]
            rec_dem = output[ChannelEnum.REC_DEM]
            inpainted_elevation_map = output[ChannelEnum.COMP_DEM]

            # features for the ground-truth, the reconstructed elevation map and the inpainted elevation map
            # the feature extractor expects an image with three channels as an input
            feat_gt = self.feature_extractor(gt_dem.unsqueeze(dim=1).repeat(1, 3, 1, 1))
            feat_recons = self.feature_extractor(rec_dem.unsqueeze(dim=1).repeat(1, 3, 1, 1))
            feat_inpaint = self.feature_extractor(inpainted_elevation_map.unsqueeze(dim=1).repeat(1, 3, 1, 1))

            for i in range(len(feat_gt)):
                perceptual_loss += perceptual_loss_fct(feat_recons[i], feat_gt[i], **kwargs)
                perceptual_loss += perceptual_loss_fct(feat_inpaint[i], feat_gt[i], **kwargs)

                style_loss += style_loss_fct(feat_recons[i], feat_gt[i], **kwargs)
                style_loss += style_loss_fct(feat_inpaint[i], feat_gt[i], **kwargs)

        return {LossEnum.PERCEPTUAL: perceptual_loss,
                LossEnum.STYLE: style_loss}
