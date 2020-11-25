from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.nn import functional as F
from typing import *

from src.enums import *
from src.datasets.base_dataset import BaseDataset
from src.learning.loss.loss import masked_loss_fct, mse_loss_fct, \
    l1_loss_fct, psnr_loss_fct, ssim_loss_fct, perceptual_loss_fct, style_loss_fct
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

        if input_normalization is None or input_normalization is False:
            self.input_normalization = None
        else:
            self.input_normalization = input_normalization

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass

    def assemble_input(self, data: Dict[Union[str, ChannelEnum], torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        input = None
        norm_consts = {}
        for channel_idx, in_channel in enumerate(self.in_channels):
            if in_channel in data:
                channel_data = data[in_channel]
            else:
                raise NotImplementedError

            if self.input_normalization is not None:
                if in_channel == ChannelEnum.OCCLUDED_ELEVATION_MAP or \
                        in_channel == ChannelEnum.GROUND_TRUTH_ELEVATION_MAP:
                    channel_data, norm_consts[in_channel] = InputNormalization.normalize(in_channel, channel_data,
                                                                                         **self.input_normalization,
                                                                                         batch=True)

            if in_channel == ChannelEnum.OCCLUDED_ELEVATION_MAP:
                channel_data = self.preprocess_occluded_elevation_map(channel_data)

            if in_channel == ChannelEnum.BINARY_OCCLUSION_MAP:
                channel_data = ~channel_data

            if input is None:
                input_size = (channel_data.size(0), len(self.in_channels), channel_data.size(1), channel_data.size(2))
                input = channel_data.new_zeros(size=input_size, dtype=torch.float32)

            input[:, channel_idx, ...] = channel_data.unsqueeze(dim=1)[:, 0, :, :]

        return input, norm_consts

    def preprocess_occluded_elevation_map(self, occluded_elevation_map: torch.Tensor) -> torch.Tensor:
        poem = occluded_elevation_map.clone()

        NaN_replacement = self.config.get("NaN_replacement", 0)

        # replace NaNs signifying occluded areas with arbitrary high or low number
        # poem[occluded_elevation_map != occluded_elevation_map] = -10000
        poem[occluded_elevation_map != occluded_elevation_map] = NaN_replacement

        # TODO: for torch==1.8
        # poem = torch.nan_to_num(occluded_elevation_map, nan=NaN_replacement)

        return poem

    def create_inpainted_elevation_map(self, occluded_elevation_map: torch.Tensor,
                                       reconstructed_elevation_map: torch.Tensor) -> torch.Tensor:
        inpainted_elevation_map = occluded_elevation_map.clone()

        selector = torch.isnan(occluded_elevation_map)
        inpainted_elevation_map[selector] = reconstructed_elevation_map[selector]

        return inpainted_elevation_map

    def denormalize_output(self,
                           data: Dict[ChannelEnum, torch.Tensor],
                           output: Dict[Union[ChannelEnum, str], torch.Tensor],
                           norm_consts: dict) -> Dict[Union[ChannelEnum, str], torch.Tensor]:

        if self.input_normalization is not None:
            denormalized_output = {}
            for key, value in output.items():
                if ChannelEnum.RECONSTRUCTED_ELEVATION_MAP:
                    if ChannelEnum.GROUND_TRUTH_ELEVATION_MAP in norm_consts:
                        denormalize_norm_const = norm_consts[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP]
                    elif ChannelEnum.OCCLUDED_ELEVATION_MAP in norm_consts:
                        denormalize_norm_const = norm_consts[ChannelEnum.OCCLUDED_ELEVATION_MAP]
                    else:
                        raise ValueError

                    denormalized_output[key] = InputNormalization.denormalize(ChannelEnum.RECONSTRUCTED_ELEVATION_MAP,
                                                                              input=value, batch=True,
                                                                              norm_consts=denormalize_norm_const,
                                                                              **self.input_normalization)
                else:
                    denormalized_output[key] = value
        else:
            denormalized_output = output

        reconstructed_elevation_map = denormalized_output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP]
        inpainted_elevation_map = self.create_inpainted_elevation_map(data[ChannelEnum.OCCLUDED_ELEVATION_MAP],
                                                                      reconstructed_elevation_map)
        denormalized_output[ChannelEnum.COMPOSED_ELEVATION_MAP] = inpainted_elevation_map

        return denormalized_output

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
            elevation_map = data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP]
            reconstructed_elevation_map = output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP]
            norm_elevation_map, ground_truth_norm_consts = InputNormalization.normalize(
                ChannelEnum.GROUND_TRUTH_ELEVATION_MAP,
                input=elevation_map,
                batch=True,
                mean=True, stdev=True)
            norm_reconstructed_elevation_map, _ = InputNormalization.normalize(ChannelEnum.RECONSTRUCTED_ELEVATION_MAP,
                                                                               input=reconstructed_elevation_map,
                                                                               batch=True, mean=True, stdev=True,
                                                                               norm_consts=ground_truth_norm_consts)
            norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP] = norm_elevation_map
            norm_data[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP] = norm_reconstructed_elevation_map

        return norm_data

    def eval_loss_function(self,
                           loss_config: dict,
                           output: Dict[Union[ChannelEnum, str], torch.Tensor],
                           data: Dict[ChannelEnum, torch.Tensor],
                           dataset: BaseDataset = None,
                           **kwargs) -> dict:
        sample_tensor = output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP]

        loss_dict = {}

        if ChannelEnum.GROUND_TRUTH_ELEVATION_MAP not in data:
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
            loss_dict[LossEnum.L1_REC_ALL] = l1_loss_fct(norm_data[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                                                         norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP], **kwargs)
        else:
            loss_dict[LossEnum.L1_REC_ALL] = l1_loss_fct(output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                                                         data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP], **kwargs)

        if LossEnum.L1_REC_OCC in norm_data:
            loss_dict[LossEnum.L1_REC_OCC] = masked_loss_fct(
                l1_loss_fct,
                norm_data[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                data[ChannelEnum.BINARY_OCCLUSION_MAP],
                **kwargs)
        else:
            loss_dict[LossEnum.L1_REC_OCC] = masked_loss_fct(
                l1_loss_fct,
                output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                data[ChannelEnum.BINARY_OCCLUSION_MAP],
                **kwargs)

        if LossEnum.L1_REC_NOCC in norm_data:
            loss_dict[LossEnum.L1_REC_NOCC] = masked_loss_fct(
                l1_loss_fct,
                norm_data[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                ~data[ChannelEnum.BINARY_OCCLUSION_MAP],
                **kwargs)
        else:
            loss_dict[LossEnum.L1_REC_NOCC] = masked_loss_fct(
                l1_loss_fct,
                output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                ~data[ChannelEnum.BINARY_OCCLUSION_MAP],
                **kwargs)

        if LossEnum.L1_COMP_ALL in norm_data:
            loss_dict[LossEnum.L1_COMP_ALL] = l1_loss_fct(norm_data[ChannelEnum.COMPOSED_ELEVATION_MAP],
                                                          norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP], **kwargs)
        else:
            loss_dict[LossEnum.L1_COMP_ALL] = l1_loss_fct(output[ChannelEnum.COMPOSED_ELEVATION_MAP],
                                                          data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP], **kwargs)

        if LossEnum.MSE_REC_ALL in norm_data:
            loss_dict[LossEnum.MSE_REC_ALL] = mse_loss_fct(norm_data[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                                                           norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP], **kwargs)
        else:
            loss_dict[LossEnum.MSE_REC_ALL] = mse_loss_fct(output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                                                           data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP], **kwargs)

        if LossEnum.MSE_REC_OCC in norm_data:
            loss_dict[LossEnum.MSE_REC_OCC] = masked_loss_fct(
                mse_loss_fct,
                norm_data[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                data[ChannelEnum.BINARY_OCCLUSION_MAP],
                **kwargs)
        else:
            loss_dict[LossEnum.MSE_REC_OCC] = masked_loss_fct(
                mse_loss_fct,
                output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                data[ChannelEnum.BINARY_OCCLUSION_MAP],
                **kwargs)

        if LossEnum.MSE_REC_NOCC in norm_data:
            loss_dict[LossEnum.MSE_REC_NOCC] = masked_loss_fct(
                mse_loss_fct,
                norm_data[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                ~data[ChannelEnum.BINARY_OCCLUSION_MAP],
                **kwargs)
        else:
            loss_dict[LossEnum.MSE_REC_NOCC] = masked_loss_fct(
                mse_loss_fct,
                output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                ~data[ChannelEnum.BINARY_OCCLUSION_MAP],
                **kwargs)

        if LossEnum.MSE_COMP_ALL in norm_data:
            loss_dict[LossEnum.MSE_COMP_ALL] = mse_loss_fct(norm_data[ChannelEnum.COMPOSED_ELEVATION_MAP],
                                                            norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP], **kwargs)
        else:
            loss_dict[LossEnum.MSE_COMP_ALL] = mse_loss_fct(output[ChannelEnum.COMPOSED_ELEVATION_MAP],
                                                            data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP], **kwargs)

        if LossEnum.SSIM_REC in norm_data:
            loss_dict[LossEnum.SSIM_REC] = ssim_loss_fct(norm_data[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                                                         norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                                                         data_min=dataset.min, data_max=dataset.max,
                                                         **kwargs)
        else:
            loss_dict[LossEnum.SSIM_REC] = ssim_loss_fct(output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                                                         data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                                                         data_min=dataset.min, data_max=dataset.max,
                                                         **kwargs)
        if LossEnum.SSIM_COMP in norm_data:
            loss_dict[LossEnum.SSIM_COMP] = ssim_loss_fct(norm_data[ChannelEnum.COMPOSED_ELEVATION_MAP],
                                                          norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                                                          data_min=dataset.min, data_max=dataset.max,
                                                          **kwargs)
        else:
            loss_dict[LossEnum.SSIM_COMP] = ssim_loss_fct(output[ChannelEnum.COMPOSED_ELEVATION_MAP],
                                                          data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                                                          data_min=dataset.min, data_max=dataset.max,
                                                          **kwargs)

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
            ground_truth_elevation_map = data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP]
            reconstructed_elevation_map = output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP]
            inpainted_elevation_map = output[ChannelEnum.COMPOSED_ELEVATION_MAP]

            # features for the ground-truth, the reconstructed elevation map and the inpainted elevation map
            # the feature extractor expects an image with three channels as an input
            feat_gt = self.feature_extractor(ground_truth_elevation_map.unsqueeze(dim=1).repeat(1, 3, 1, 1))
            feat_recons = self.feature_extractor(reconstructed_elevation_map.unsqueeze(dim=1).repeat(1, 3, 1, 1))
            feat_inpaint = self.feature_extractor(inpainted_elevation_map.unsqueeze(dim=1).repeat(1, 3, 1, 1))

            for i in range(len(feat_gt)):
                perceptual_loss += perceptual_loss_fct(feat_recons[i], feat_gt[i], **kwargs)
                perceptual_loss += perceptual_loss_fct(feat_inpaint[i], feat_gt[i], **kwargs)

                style_loss += style_loss_fct(feat_recons[i], feat_gt[i], **kwargs)
                style_loss += style_loss_fct(feat_inpaint[i], feat_gt[i], **kwargs)

        return {LossEnum.PERCEPTUAL: perceptual_loss,
                LossEnum.STYLE: style_loss}
