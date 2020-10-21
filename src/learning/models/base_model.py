from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.nn import functional as F
from typing import *

from src.enums import *
from src.learning.loss.loss import mse_loss_fct, reconstruction_occlusion_loss_fct
from src.learning.normalization.input_normalization import InputNormalization


class BaseModel(ABC, nn.Module):
    def __init__(self, in_channels: List[str], out_channels: List[str],
                 input_normalization: Dict = None, **kwargs):
        super().__init__()
        self.config = kwargs

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
        denormalized_output[ChannelEnum.INPAINTED_ELEVATION_MAP] = inpainted_elevation_map

        return denormalized_output

    def get_normalized_data(self,
                            loss_config: dict,
                            output: Dict[Union[ChannelEnum, str], torch.Tensor],
                            data: Dict[ChannelEnum, torch.Tensor],
                            **kwargs) -> Dict[ChannelEnum, torch.Tensor]:
        normalization_config = loss_config.get("normalization", [])
        norm_data = {}
        if LossEnum.RECONSTRUCTION.value in normalization_config or \
                LossEnum.RECONSTRUCTION_OCCLUSION.value in normalization_config or \
                LossEnum.RECONSTRUCTION_NON_OCCLUSION.value in normalization_config:
            elevation_map = data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP]
            reconstructed_elevation_map = output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP]
            norm_elevation_map, ground_truth_norm_consts = InputNormalization.normalize(ChannelEnum.GROUND_TRUTH_ELEVATION_MAP,
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
                           **kwargs) -> dict:
        sample_tensor = output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP]

        if ChannelEnum.GROUND_TRUTH_ELEVATION_MAP not in data:
            return {LossEnum.LOSS: sample_tensor.new_tensor(0),
                    LossEnum.RECONSTRUCTION: sample_tensor.new_tensor(0),
                    LossEnum.RECONSTRUCTION_OCCLUSION: sample_tensor.new_tensor(0),
                    LossEnum.RECONSTRUCTION_NON_OCCLUSION: sample_tensor.new_tensor(0),
                    LossEnum.INPAINTING: sample_tensor.new_tensor(0)}

        norm_data = self.get_normalized_data(loss_config, output, data, **kwargs)

        if LossEnum.RECONSTRUCTION in norm_data:
            recons_loss = mse_loss_fct(norm_data[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                                       norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP], **kwargs)
        else:
            recons_loss = mse_loss_fct(output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                                       data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP], **kwargs)

        if LossEnum.INPAINTING in norm_data:
            inpainting_loss = mse_loss_fct(norm_data[ChannelEnum.INPAINTED_ELEVATION_MAP],
                                           norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP], **kwargs)
        else:
            inpainting_loss = mse_loss_fct(output[ChannelEnum.INPAINTED_ELEVATION_MAP],
                                           data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP], **kwargs)

        if LossEnum.RECONSTRUCTION_OCCLUSION in norm_data:
            recons_occlusion_loss = reconstruction_occlusion_loss_fct(
                norm_data[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                data[ChannelEnum.BINARY_OCCLUSION_MAP],
                **kwargs)
        else:
            recons_occlusion_loss = reconstruction_occlusion_loss_fct(output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                                                                      data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                                                                      data[ChannelEnum.BINARY_OCCLUSION_MAP],
                                                                      **kwargs)

        if LossEnum.RECONSTRUCTION_NON_OCCLUSION in norm_data:
            recons_non_occlusion_loss = reconstruction_occlusion_loss_fct(
                norm_data[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                norm_data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                ~data[ChannelEnum.BINARY_OCCLUSION_MAP],
                **kwargs)
        else:
            recons_non_occlusion_loss = reconstruction_occlusion_loss_fct(
                output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP],
                data[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP],
                ~data[ChannelEnum.BINARY_OCCLUSION_MAP],
                **kwargs)

        weights = loss_config.get("eval_weights", {})
        recons_weight = weights.get(LossEnum.RECONSTRUCTION.value, 0)
        recons_occlusion_weight = weights.get(LossEnum.RECONSTRUCTION_OCCLUSION.value, 1)
        recons_non_occlusion_weight = weights.get(LossEnum.RECONSTRUCTION_NON_OCCLUSION.value, 0)

        loss = recons_weight * recons_loss + \
               recons_occlusion_weight * recons_occlusion_loss + \
               recons_non_occlusion_weight * recons_non_occlusion_loss

        return {LossEnum.LOSS: loss,
                LossEnum.RECONSTRUCTION: recons_loss,
                LossEnum.RECONSTRUCTION_OCCLUSION: recons_occlusion_loss,
                LossEnum.RECONSTRUCTION_NON_OCCLUSION: recons_non_occlusion_loss,
                LossEnum.INPAINTING: inpainting_loss}

    def artistic_loss_function(self,
                               loss_config: dict,
                               output: Dict[Union[ChannelEnum, str], torch.Tensor],
                               data: Dict[ChannelEnum, torch.Tensor],
                               **kwargs) -> dict:
        # Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge.
        # "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).
        # https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/loss.py
        # this requires a self.feature_extractor

        # the feature extractor expects an image with three channels as an input

        # TODO: this needs to get finished
        # feat_recons = self.feature_extractor(output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP])
        # feat_inpaint = self.feature_extractor(output[ChannelEnum.INPAINTED_ELEVATION_MAP])

        perceptual_loss = 0
        style_loss = 0

        return {LossEnum.PERCEPTUAL: perceptual_loss,
                LossEnum.STYLE: style_loss}
