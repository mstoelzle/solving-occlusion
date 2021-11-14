import numpy as np
from scipy.linalg import lstsq
import torch
from typing import *

from src.enums import *
from .base_baseline_model import BaseBaselineModel


# @torch.jit.script
def lsq_plane_fit(occ_dem: torch.Tensor, occ_mask: torch.Tensor, max_occ_ratio: float = 0.75):
    rec_dem = occ_dem.clone()
    occ_indices = torch.nonzero(occ_mask == 1)

    for occ_cell_idx in range(occ_indices.size(0)):
        target_cell_indice = occ_indices[occ_cell_idx, ...]
        print("target_cell_indice", target_cell_indice)

        occ_ratio = 1.0
        pixel_radius = 1
        patch_dem, patch_mask, target_patch_cell_indice = None, None, None
        while occ_ratio > max_occ_ratio:
            pixel_radius += 1

            start_idx_x = max(int(target_cell_indice[0] - pixel_radius), 0)
            start_idx_y = max(int(target_cell_indice[1] - pixel_radius), 0)
            stop_idx_x = min(int(target_cell_indice[0] + pixel_radius), occ_dem.size(0) - 1)
            stop_idx_y = min(int(target_cell_indice[1] + pixel_radius), occ_dem.size(1) - 1)

            patch_dem = occ_dem[start_idx_x:stop_idx_x, start_idx_y:stop_idx_y]
            patch_mask = occ_mask[start_idx_x:stop_idx_x, start_idx_y:stop_idx_y]

            target_patch_cell_indice = target_cell_indice - torch.tensor([start_idx_x, start_idx_y])

            if target_cell_indice[0] == 33 and target_cell_indice[1] == 44:
                print(patch_dem)
                print(patch_mask)
                print(target_patch_cell_indice)
            # print("target_patch_cell_indice", target_patch_cell_indice, "patch_dem", patch_dem)

            occ_ratio = torch.sum(patch_mask == 1) / ((stop_idx_x - start_idx_x)*(stop_idx_y - start_idx_y))

        patch_nocc_indices = torch.nonzero(patch_mask == 0)
        patch_nocc_z = patch_dem[patch_mask == 0]

        # should be 3xn
        A = torch.cat([patch_nocc_indices, torch.ones(size=(patch_nocc_z.size(0), 1))], dim=1)
        b = patch_nocc_z.unsqueeze(1)  # should be 1xn
        # print("A shape", A.size(), "b shape", b.size())

        fit, residual, rnk, s = lstsq(A, b)
        # print("fit", fit, "residual", residual, "rnk", rnk, "s", s)

        patch_x, patch_y = target_patch_cell_indice[0].item(), target_patch_cell_indice[1].item()
        fitted_z_value = (fit[0] * patch_x + fit[1] * patch_y + fit[2]).item()

        rec_dem[target_cell_indice[0], target_cell_indice[1]] = fitted_z_value

        print("fitted z-value", fitted_z_value)

    return rec_dem


class LsqPlaneFitBaseline(BaseBaselineModel):
    def __init__(self, max_occ_ratio_plane_fit: float = 0.75, **kwargs):
        super().__init__(**kwargs)

        self.max_occ_ratio_plane_fit = max_occ_ratio_plane_fit

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

            rec_dem = lsq_plane_fit(occ_dem, occ_mask, self.max_occ_ratio_plane_fit)

            if ChannelEnum.GT_DEM in data:
                gt_dem = data[ChannelEnum.GT_DEM][idx, ...]
            else:
                gt_dem = None

            import matplotlib.pyplot as plt
            h, w = occ_dem.size(0), occ_dem.size(1)
            plt.subplot(221)
            plt.imshow(occ_dem.cpu().numpy().T, extent=(0, w-1, 0, h-1), origin='lower')
            plt.title('Occluded DEM')
            plt.subplot(222)
            plt.imshow(occ_mask.cpu().numpy().T, extent=(0, w-1, 0, h-1), origin='lower')
            plt.title('Occlusion mask')
            plt.subplot(223)
            plt.imshow(rec_dem.cpu().numpy().T, extent=(0, w-1, 0, h-1), origin='lower')
            plt.title('Reconstructed DEM')
            plt.subplot(224)
            plt.imshow(gt_dem.cpu().numpy().T, extent=(0, w-1, 0, h-1), origin='lower')
            plt.title('Ground-truth DEM')
            plt.gcf().set_size_inches(6, 6)
            plt.show()

            exit()

            rec_dems[idx, ...] = rec_dem

        return rec_dems

    def loss_function(self, **kwargs) -> dict:
        return self.eval_loss_function(**kwargs)
