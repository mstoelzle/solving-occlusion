import torch
from typing import *

from src.enums import *
from src.learning.models.base_model import BaseModel


class BaseBaselineModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(strict_forward_def=False, **kwargs)

    def forward_pass(self, data: Dict[Union[ChannelEnum, str], torch.Tensor],
                     **kwargs) -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        output = {}

        assert(self.strict_forward_def is False)
        x = self.forward(data=data)

        rec_dem = x.squeeze(dim=1)
        output[ChannelEnum.REC_DEM] = rec_dem

        comp_dem = self.create_composed_map(data[ChannelEnum.OCC_DEM], rec_dem)

        output = {ChannelEnum.REC_DEM: rec_dem,
                  ChannelEnum.COMP_DEM: comp_dem}

        return output
