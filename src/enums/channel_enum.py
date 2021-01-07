from aenum import MultiValueEnum


class ChannelEnum(MultiValueEnum):
    PARAMS = "params"  # deprecated - use RES_GRID and REL_POSE instead
    RES_GRID = "resolution_grid"  # resolution of grid in x and y dir
    REL_POSITION = "relative_position"  # relative position of robot to center of grid (3D)
    # relative attitude of robot to grid in quarternion (4D). quaternion in scalar-last (x, y, z, w)
    REL_ATTITUDE = "relative_attitude"
    GT_DEM = "gt_dem", "ground_truth_elevation_map", "elevation_map"
    OCC_DEM = "occ_dem", "occluded_elevation_map"
    OCC_MASK = "occ_mask", "binary_occlusion_map"
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
