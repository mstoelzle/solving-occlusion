from enum import Enum


class ChannelEnum(Enum):
    PARAMS = "params"
    GT_DEM = "gt_dem"
    OCC_DEM = "occ_dem"
    OCC_MASK = "occ_mask"
    REC_DEM = "rec_dem"
    REC_DEMS = "rec_dems"
    COMP_DEM = "comp_dem"
    COMP_DEMS = "comp_dems"
    OCC_DATA_UNCERTAINTY_MAP = "occ_data_uncertainty_map"
    DATA_UNCERTAINTY_MAP = "data_uncertainty_map"
    MODEL_UNCERTAINTY_MAP = "model_uncertainty_map"
    TOTAL_UNCERTAINTY_MAP = "total_uncertainty_map"
    TRAV_RISK_MAP = "trav_risk_map"
