import torch
from typing import *

from src.enums import *
from src.learning.models.base_model import BaseModel


class BaseBaselineModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(strict_forward_def=False, **kwargs)

    def forward_pass(self, data: Dict[Union[ChannelEnum, str], torch.Tensor],
                     **kwargs) -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        input, norm_consts = self.assemble_input(data)

        output = {}

        if self.strict_forward_def:
            x = self.forward(input=input)
        else:
            x = self.forward(input=input, data=data)

        x = x.squeeze(dim=1)
        rec_dem = x
        output[ChannelEnum.REC_DEM] = rec_dem

        output = self.denormalize_output(data, output, norm_consts)

        return output
