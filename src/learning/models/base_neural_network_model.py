import torch
from typing import *

from src.enums import *
from .base_model import BaseModel


class BaseNeuralNetworkModel(BaseModel):
    def forward_pass(self, data: Dict[Union[ChannelEnum, str], torch.Tensor],
                     **kwargs) -> Dict[Union[ChannelEnum, str], torch.Tensor]:
        input, norm_consts = self.assemble_input(data)

        output = {}

        self.set_dropout_mode(dropout_mode=True if self.training and self.use_training_dropout else False)

        data_uncertainty = None

        if self.strict_forward_def:
            x = self.forward(input=input)
        else:
            x = self.forward(input=input, data=data)

        if type(x) in [list, tuple]:
            # remove channels dimension from tensor
            for i in range(len(x)):
                x[i] = x[i].squeeze(dim=1)

            rec_dem = x[0]
            data_uncertainty = x[1]
        else:
            x = x.squeeze(dim=1)
            rec_dem = x
        output[ChannelEnum.REC_DEM] = rec_dem

        model_uncertainty = None
        if self.num_solutions > 1 and self.training is False:
            if self.model_uncertainty_method == ModelUncertaintyMethodEnum.MONTE_CARLO_DROPOUT:
                self.set_dropout_mode(True)

                dem_solutions = []
                data_uncertainties = []
                for i in range(self.num_solutions):
                    x = self.forward(input=input)

                    if type(x) in [list, tuple]:
                        # remove channels dimension from tensor
                        for i in range(len(x)):
                            x[i] = x[i].squeeze(dim=1)

                        dem_solutions.append(x[0])
                        data_uncertainties.append(x[1])
                    else:
                        x = x.squeeze(dim=1)
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

            output[ChannelEnum.MODEL_UM] = model_uncertainty
            output[ChannelEnum.REC_DEMS] = dem_solutions

        if data_uncertainty is not None:
            output[ChannelEnum.REC_DATA_UM] = data_uncertainty
        if model_uncertainty is not None:
            output[ChannelEnum.MODEL_UM] = model_uncertainty

        output = self.denormalize_output(data, output, norm_consts)

        return output
