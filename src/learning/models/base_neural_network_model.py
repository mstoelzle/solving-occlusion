from copy import deepcopy
from distutils.version import StrictVersion
import torch
from typing import *

from src.enums import *
from .base_model import BaseModel
from .baseline.open_cv_baseline import OpenCVBaseline
from .baseline.interpolation_baseline import InterpolationBaseline
from src.learning.normalization.input_normalization import InputNormalization


class BaseNeuralNetworkModel(BaseModel):
    def __init__(self, seed: int, in_channels: List[str], out_channels: List[str], input_normalization: Dict = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.seed = seed

        self.in_channels = [ChannelEnum(in_channel) for in_channel in in_channels]
        self.out_channels = [ChannelEnum(out_channel) for out_channel in out_channels]

        self.input_normalization = None if input_normalization is False else input_normalization

        self.NaN_replacement = self.config.get("NaN_replacement", 0.0)

        self.NaN_replacement_model = None
        if isinstance(self.NaN_replacement, str) or isinstance(self.NaN_replacement, dict):
            baseline_models = {
                "cubic": InterpolationBaseline,
                "linear": InterpolationBaseline,
                "NavierStokes": OpenCVBaseline,
                "nearest": InterpolationBaseline,
                "Telea": OpenCVBaseline,
            }

            if isinstance(self.NaN_replacement, str):
                # use the same model config as for the NN
                replacement_model_config = deepcopy(kwargs)
                replacement_model_config["name"] = self.NaN_replacement
            else:
                # use the model config specified in the NaN_replacement dictionary
                replacement_model_config = self.NaN_replacement

            self.NaN_replacement_model = baseline_models[replacement_model_config["name"]](**replacement_model_config)

        self.dropout_mode = False
        self.dropout_p = self.config.get("training_dropout_probability", 0.0)
        self.use_training_dropout = True if self.dropout_p > 0.0 else False

        self.adf = False
        self.keep_variance_fn = None
        if self.config.get("data_uncertainty_estimation") is not None:
            data_uncertainty_config = self.config["data_uncertainty_estimation"]
            self.data_uncertainty_method = DataUncertaintyMethodEnum(data_uncertainty_config["method"])
            if self.data_uncertainty_method == DataUncertaintyMethodEnum.ADF:
                self.adf = True
                min_variance = data_uncertainty_config.get("min_variance", 0.001)
                self.keep_variance_fn = lambda x: adf.keep_variance(x, min_variance=min_variance)
            else:
                raise NotImplementedError

        self.model_uncertainty_method = None
        self.num_solutions: int = 1
        if self.config.get("model_uncertainty_estimation") is not None:
            model_uncertainty_config = self.config["model_uncertainty_estimation"]
            self.model_uncertainty_method = ModelUncertaintyMethodEnum(model_uncertainty_config["method"])
            if self.model_uncertainty_method == ModelUncertaintyMethodEnum.MONTE_CARLO_DROPOUT:
                if self.use_training_dropout:
                    assert self.dropout_p == model_uncertainty_config["probability"]
                else:
                    self.dropout_p = model_uncertainty_config["probability"]

                self.num_solutions = int(model_uncertainty_config["num_solutions"])
                self.use_mean_as_rec = model_uncertainty_config.get("use_mean_as_rec", False)
            elif self.model_uncertainty_method == ModelUncertaintyMethodEnum.MONTE_CARLO_VAE:
                self.num_solutions = int(model_uncertainty_config["num_solutions"])
                self.use_mean_as_rec = model_uncertainty_config.get("use_mean_as_rec", False)
            else:
                raise NotImplementedError

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

    def assemble_input(self, data: Dict[Union[str, ChannelEnum], torch.Tensor]) \
            -> Tuple[Union[List, torch.Tensor], Dict]:
        input = None
        var = None
        norm_consts = {}
        for channel_idx, in_channel in enumerate(self.in_channels):
            if in_channel in data:
                channel_data = data[in_channel]
            else:
                raise NotImplementedError

            if self.input_normalization is not None:
                if in_channel == ChannelEnum.OCC_DEM or \
                        in_channel == ChannelEnum.GT_DEM:
                    channel_data, norm_consts[in_channel] = InputNormalization.normalize(in_channel, channel_data,
                                                                                         **self.input_normalization,
                                                                                         batch=True)

            if in_channel == ChannelEnum.OCC_DEM:
                channel_data = self.preprocess_occluded_map(channel_data,
                                                            NaN_replacement=self.NaN_replacement,
                                                            data=data)
            elif in_channel == ChannelEnum.OCC_MASK:
                channel_data = ~channel_data

            if input is None:
                input_size = (channel_data.size(0), len(self.in_channels), channel_data.size(1), channel_data.size(2))
                input = channel_data.new_zeros(size=input_size, dtype=torch.float32)

                if self.adf:
                    var = channel_data.new_zeros(size=input_size, dtype=torch.float32)

            input[:, channel_idx, ...] = channel_data

            if self.adf:
                if in_channel == ChannelEnum.OCC_DEM:
                    occ_data_um = data[ChannelEnum.OCC_DATA_UM]

                    if "NaN_replacement" in self.config.get("data_uncertainty_estimation", {}):
                        NaN_replacement = self.config["data_uncertainty_estimation"]["NaN_replacement"]
                    else:
                        NaN_replacement = self.config.get("NaN_replacement", 0.0)

                    occ_data_um = self.preprocess_occluded_map(occ_data_um, NaN_replacement=NaN_replacement, data=data)
                    var[:, channel_idx, ...] = occ_data_um

        if self.adf:
            input = [input, var]

        return input, norm_consts

    def preprocess_occluded_map(self, occ_dem: torch.Tensor, NaN_replacement: Union[float, int, str] = 0.0,
                                data: dict = None) -> torch.Tensor:
        if self.NaN_replacement_model is not None:
            assert(data is not None)
            output = self.NaN_replacement_model.forward_pass(data)
            poem = output[ChannelEnum.COMP_DEM]
        else:
            if StrictVersion(torch.__version__) < StrictVersion("1.8.0"):
                poem = occ_dem.clone()

                # replace NaNs signifying occluded areas with arbitrary high or low number
                poem[occ_dem != occ_dem] = NaN_replacement
            else:
                poem = torch.nan_to_num(occ_dem, nan=NaN_replacement)

        return poem

    def denormalize_output(self,
                           data: Dict[ChannelEnum, torch.Tensor],
                           output: Dict[Union[ChannelEnum, str], torch.Tensor],
                           norm_consts: dict) -> Dict[Union[ChannelEnum, str], torch.Tensor]:

        if self.input_normalization is not None:
            denorm_output = {}
            for key, value in output.items():
                if key in [ChannelEnum.REC_DEM, ChannelEnum.REC_DEMS]:
                    if ChannelEnum.GT_DEM in norm_consts:
                        denormalize_norm_const = norm_consts[ChannelEnum.GT_DEM]
                    elif ChannelEnum.OCC_DEM in norm_consts:
                        denormalize_norm_const = norm_consts[ChannelEnum.OCC_DEM]
                    else:
                        raise ValueError

                    denorm_output[key] = InputNormalization.denormalize(key, input=value, batch=True,
                                                                        norm_consts=denormalize_norm_const,
                                                                        **self.input_normalization)
                else:
                    denorm_output[key] = value
        else:
            denorm_output = output

        rec_dem = denorm_output[ChannelEnum.REC_DEM]
        comp_dem = self.create_composed_map(data[ChannelEnum.OCC_DEM], rec_dem)
        denorm_output[ChannelEnum.COMP_DEM] = comp_dem

        if ChannelEnum.REC_DEMS in denorm_output:
            rec_dems = denorm_output[ChannelEnum.REC_DEMS]
            occ_dems = []
            for i in range(rec_dems.size(dim=1)):
                occ_dems.append(data[ChannelEnum.OCC_DEM])
            occ_dems = torch.stack(occ_dems, dim=1)
            denorm_output[ChannelEnum.COMP_DEMS] = self.create_composed_map(occ_dems, rec_dems)

        if ChannelEnum.OCC_DATA_UM in data and ChannelEnum.REC_DATA_UM in denorm_output:
            occ_data_um, rec_data_um = data[ChannelEnum.OCC_DATA_UM], denorm_output[ChannelEnum.REC_DATA_UM]
            denorm_output[ChannelEnum.COMP_DATA_UM] = self.create_composed_map(occ_data_um, rec_data_um)

        if ChannelEnum.COMP_DATA_UM in denorm_output and ChannelEnum.MODEL_UM in denorm_output:
            denorm_output[ChannelEnum.TOTAL_UM] = denorm_output[ChannelEnum.COMP_DATA_UM] \
                                                  + denorm_output[ChannelEnum.MODEL_UM]
        elif ChannelEnum.COMP_DATA_UM in denorm_output:
            denorm_output[ChannelEnum.TOTAL_UM] = denorm_output[ChannelEnum.COMP_DATA_UM]
        elif ChannelEnum.MODEL_UM in denorm_output:
            denorm_output[ChannelEnum.TOTAL_UM] = denorm_output[ChannelEnum.MODEL_UM]

        return denorm_output
