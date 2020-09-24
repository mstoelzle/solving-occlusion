from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.nn import functional as F
from typing import *

from src.enums import *
from src.learning.loss.loss import reconstruction_occlusion_loss_fct
from src.learning.normalization.input_normalization import InputNormalization


class BaseModel(ABC, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

        self.input_normalization: bool = self.config.get("input_normalization", True)

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass

    def assemble_input(self, data: Dict[Union[str, ChannelEnum], torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        data[ChannelEnum.BINARY_OCCLUSION_MAP] = self.create_binary_occlusion_map(data[ChannelEnum.OCCLUDED_ELEVATION_MAP])

        input = None
        norm_consts = {}
        for channel_idx, in_channel in enumerate(self.in_channels):
            if in_channel in data:
                channel_data = data[in_channel]
            else:
                raise NotImplementedError

            if self.input_normalization:
                if in_channel == ChannelEnum.OCCLUDED_ELEVATION_MAP or in_channel == ChannelEnum.ELEVATION_MAP:
                    channel_data, norm_consts[in_channel] = InputNormalization.normalize(in_channel, channel_data,
                                                                                         batch=True)

            if in_channel == ChannelEnum.OCCLUDED_ELEVATION_MAP:
                channel_data = self.preprocess_occluded_elevation_map(channel_data)

            if input is None:
                input = channel_data.new_zeros(size=(channel_data.size(0), len(self.in_channels),
                                                     channel_data.size(1), channel_data.size(2)))

            input[:, channel_idx, ...] = channel_data.unsqueeze(dim=1)[:, 0, :, :]

        return input, norm_consts

    def preprocess_occluded_elevation_map(self, occluded_elevation_map: torch.Tensor) -> torch.Tensor:
        poem = occluded_elevation_map.clone()

        # replace NaNs signifying occluded areas with arbitrary high or low number
        poem[occluded_elevation_map != occluded_elevation_map] = -10000

        return poem

    def create_binary_occlusion_map(self, occluded_elevation_map: torch.Tensor) -> torch.Tensor:
        binary_occlusion_map = (occluded_elevation_map != occluded_elevation_map)

        return binary_occlusion_map

    def denormalize_output(self, output: Dict[Union[ChannelEnum, str], torch.Tensor],
                           norm_consts: dict) -> Dict[Union[ChannelEnum, str], torch.Tensor]:

        if self.input_normalization:
            denormalized_output = {}
            for key, value in output.items():
                if self.input_normalization is True and key == ChannelEnum.RECONSTRUCTED_ELEVATION_MAP:
                    if ChannelEnum.ELEVATION_MAP in norm_consts:
                        denormalize_norm_const = norm_consts[ChannelEnum.ELEVATION_MAP]
                    elif ChannelEnum.OCCLUDED_ELEVATION_MAP in norm_consts:
                        denormalize_norm_const = norm_consts[ChannelEnum.OCCLUDED_ELEVATION_MAP]
                    else:
                        raise ValueError

                    denormalized_output[key] = InputNormalization.denormalize(ChannelEnum.RECONSTRUCTED_ELEVATION_MAP,
                                                                              input=value, batch=True,
                                                                              norm_consts=denormalize_norm_const)
                else:
                    denormalized_output[key] = value
        else:
            denormalized_output = output

        return denormalized_output

    def eval_loss_function(self,
                           loss_config: dict,
                           output: Dict[Union[ChannelEnum, str], torch.Tensor],
                           data: Dict[ChannelEnum, torch.Tensor],
                           **kwargs) -> dict:
        elevation_map = data[ChannelEnum.ELEVATION_MAP]
        reconstructed_elevation_map = output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP]

        if LossEnum.RECONSTRUCTION_OCCLUSION.value in loss_config.get("normalization", []):
            elevation_map, ground_truth_norm_consts = InputNormalization.normalize(ChannelEnum.ELEVATION_MAP,
                                                                                   input=elevation_map,
                                                                                   batch=True)
            reconstructed_elevation_map, _ = InputNormalization.normalize(ChannelEnum.RECONSTRUCTED_ELEVATION_MAP,
                                                                          input=reconstructed_elevation_map,
                                                                          batch=True,
                                                                          norm_consts=ground_truth_norm_consts)

        binary_occlusion_map = self.create_binary_occlusion_map(data[ChannelEnum.OCCLUDED_ELEVATION_MAP])

        recons_occlusion_loss = reconstruction_occlusion_loss_fct(reconstructed_elevation_map,
                                                                  elevation_map,
                                                                  binary_occlusion_map,
                                                                  **kwargs)

        recons_non_occlusion_loss = reconstruction_occlusion_loss_fct(reconstructed_elevation_map,
                                                                      elevation_map,
                                                                      ~binary_occlusion_map,
                                                                      **kwargs)

        return {LossEnum.LOSS: recons_occlusion_loss,
                LossEnum.RECONSTRUCTION_OCCLUSION: recons_occlusion_loss,
                LossEnum.RECONSTRUCTION_NON_OCCLUSION: recons_non_occlusion_loss}
