import numpy as np
import plotly.graph_objects as go
import torch
from typing import *

from src.enums import *
from src.utils.log import get_logger

logger = get_logger("live_inference_plotter")


class LiveInferencePlotter:
    def step(self, data: dict[ChannelEnum, torch.Tensor], output: dict[ChannelEnum, torch.Tensor]):
        occ_dem = data[ChannelEnum.OCC_DEM][0].detach().cpu().numpy()
        res_grid = data[ChannelEnum.RES_GRID][0].detach().cpu().numpy()
        rec_dem = output[ChannelEnum.REC_DEM][0].detach().cpu().numpy()
        comp_dem = output[ChannelEnum.COMP_DEM][0].detach().cpu().numpy()

        # start_x = -occ_dem.shape[0] // 2 * res_grid[0]
        # stop_x = occ_dem.shape[0] // 2 * res_grid[0]
        # start_y = -occ_dem.shape[1] // 2 * res_grid[1]
        # stop_y = occ_dem.shape[1] // 2 * res_grid[1]
        # x = np.arange(start=start_x, stop=stop_x, step=res_grid[0])
        # y = np.arange(start=start_y, stop=stop_y, step=res_grid[1])
        # array_y, array_x = np.meshgrid(x, y)

        fig = go.Figure(data=[go.Surface(z=occ_dem)])

        fig.update_layout(title='Mt Bruno Elevation')
        fig.show()
