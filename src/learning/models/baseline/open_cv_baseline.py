import cv2 as cv
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import *

from .base_baseline_model import BaseBaselineModel
from src.enums import *
from src.learning.normalization.input_normalization import InputNormalization


class OpenCVBaseline(BaseBaselineModel):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)

        if name == "NavierStokes":
            self.inpainting_method = cv.INPAINT_NS
        elif name == "Telea":
            self.inpainting_method = cv.INPAINT_TELEA
        elif name == "PatchMatch":
            self.inpainting_method = name
        else:
            raise ValueError

        self.inpaint_radius = self.config["inpaint_radius"]

    def forward(self, data: Dict[Union[str, ChannelEnum], torch.Tensor],
                **kwargs) -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        reconstructed_elevation_map = data[ChannelEnum.OCCLUDED_ELEVATION_MAP].clone()

        for idx in range(reconstructed_elevation_map.size(0)):
            map = data[ChannelEnum.OCCLUDED_ELEVATION_MAP][idx, ...].clone()
            binary_occlusion_map = self.create_binary_occlusion_map(map)

            min = torch.min(map[~torch.isnan(map)])
            max = torch.max(map[~torch.isnan(map)])

            map[~torch.isnan(map)] = torch.mul(map[~torch.isnan(map)] - min, 255 / (max - min))
            map[torch.isnan(map)] = 0

            np_map = map.detach().cpu().numpy()
            np_mask = binary_occlusion_map.detach().cpu().numpy().astype('uint8')

            if self.inpainting_method == "PatchMatch":
                # we are forced to use 3 channels for the PyPatchMatch package
                np_three_channel_map = np.zeros(shape=(np_mask.shape[0], np_mask.shape[1], 3), dtype="uint8")
                for channel_idx in range(3):
                    np_three_channel_map[:, :, channel_idx] = np_map

                from .py_patch_match import patch_match
                np_three_channel_reconstructed_map = patch_match.inpaint(np_three_channel_map, np_mask,
                                                                         patch_size=self.inpaint_radius)

                np_reconstructed_map = np_three_channel_reconstructed_map[:, :, 0]
            else:
                np_reconstructed_map = cv.inpaint(np_map, np_mask, self.inpaint_radius, self.inpainting_method)

            reconstructed_map = map.new_tensor(data=np_reconstructed_map)

            reconstructed_map = torch.mul(reconstructed_map,  (max - min) / 255) + min

            # import matplotlib.pyplot as plt
            # plt.matshow(np_map)
            # plt.show()
            # plt.matshow(np_mask)
            # plt.show()
            # plt.matshow(reconstructed_map)
            # plt.show()

            reconstructed_elevation_map[idx, ...] = reconstructed_map

        output = {ChannelEnum.RECONSTRUCTED_ELEVATION_MAP: reconstructed_elevation_map}
        return output

    def loss_function(self,
                      config: dict,
                      output: Dict[Union[ChannelEnum, str], torch.Tensor],
                      data: Dict[ChannelEnum, torch.Tensor],
                      **kwargs) -> dict:

        elevation_map = data[ChannelEnum.ELEVATION_MAP]
        reconstructed_elevation_map = output[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP]

        if LossEnum.RECONSTRUCTION.value in config.get("normalization", []):
            elevation_map, ground_truth_norm_consts = InputNormalization.normalize(ChannelEnum.ELEVATION_MAP,
                                                                                   input=elevation_map,
                                                                                   batch=True)
            reconstructed_elevation_map, _ = InputNormalization.normalize(ChannelEnum.RECONSTRUCTED_ELEVATION_MAP,
                                                                          input=reconstructed_elevation_map,
                                                                          batch=True,
                                                                          norm_consts=ground_truth_norm_consts)

        recons_loss = F.mse_loss(reconstructed_elevation_map, elevation_map)

        return {LossEnum.LOSS: recons_loss, LossEnum.RECONSTRUCTION: recons_loss}
