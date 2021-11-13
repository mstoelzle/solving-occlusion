import numpy as np
from scipy import interpolate
import torch
from torch import nn
from torch.nn import functional as F
from typing import *

from .base_baseline_model import BaseBaselineModel
from src.dataloaders.dataloader_meta_info import DataloaderMetaInfo
from src.enums import *
from src.learning.normalization.input_normalization import InputNormalization


class InterpolationBaseline(BaseBaselineModel):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = []
        self.out_channels = [ChannelEnum.REC_DEM]

        self.method = name

    def forward(self, data: Dict[Union[str, ChannelEnum], torch.Tensor],
                **kwargs) -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        rec_dems = data[ChannelEnum.OCC_DEM].clone()

        for idx in range(rec_dems.size(0)):
            occ_dem = data[ChannelEnum.OCC_DEM][idx, ...].clone()
            occ_mask = data[ChannelEnum.OCC_MASK][idx, ...]

            if torch.isnan(occ_dem).all():
                # the occ_dem is fully occluded
                rec_dems[idx, ...] = torch.zeros(size=occ_dem.size())
                continue

            np_occ_dem = occ_dem.detach().cpu().numpy()
            np_occ_mask = occ_mask.detach().cpu().numpy()

            points = np.argwhere(np_occ_mask == 0)
            values = np_occ_dem[np.where(np_occ_mask == 0)]

            grid_x, grid_y = np.mgrid[0:np_occ_dem.shape[0], 0:np_occ_dem.shape[1]]

            np_rec_dem = interpolate.griddata(points, values, (grid_x, grid_y), method=self.method)

            # the linear and cubic interpolation methods do not fill all holes
            # (outside of the convex hull of input points)
            if np.isnan(np_rec_dem).any():
                points = np.argwhere(np.isnan(np_rec_dem) == 0)
                values = np_rec_dem[np.where(np.isnan(np_rec_dem) == 0)]
                np_rec_dem = interpolate.griddata(points, values, (grid_x, grid_y), method="nearest")

            rec_dem = occ_dem.new_tensor(data=np_rec_dem)

            # if ChannelEnum.GT_DEM in data:
            #     gt_dem = data[ChannelEnum.GT_DEM][idx, ...]
            #     np_gt_dem = gt_dem.detach().cpu().numpy()
            # else:
            #     np_gt_dem = np.zeros(shape=np_occ_dem.shape)
            #
            # import matplotlib.pyplot as plt
            # plt.subplot(221)
            # plt.imshow(np_occ_dem.T, extent=(0, 1, 0, 1), origin='lower')
            # plt.title('Occluded DEM')
            # plt.subplot(222)
            # plt.imshow(np_occ_mask.T, extent=(0, 1, 0, 1), origin='lower')
            # plt.title('Occlusion mask')
            # plt.subplot(223)
            # plt.imshow(np_rec_dem.T, extent=(0, 1, 0, 1), origin='lower')
            # plt.title('Reconstructed DEM')
            # plt.subplot(224)
            # plt.imshow(np_gt_dem.T, extent=(0, 1, 0, 1), origin='lower')
            # plt.title('Ground-truth DEM')
            # plt.gcf().set_size_inches(6, 6)
            # plt.show()

            rec_dems[idx, ...] = rec_dem

        return rec_dems

    def loss_function(self, **kwargs) -> dict:
        return self.eval_loss_function(**kwargs)
