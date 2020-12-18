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
    OCC_DATA_UM = "occ_data_um"
    REC_DATA_UM = "rec_data_um"
    COMP_DATA_UM = "comp_data_um"
    MODEL_UM = "model_um"
    TOTAL_UM = "total_um"
    TRAV_RISK_MAP = "trav_risk_map"
