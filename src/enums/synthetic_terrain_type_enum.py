from enum import IntEnum

class SyntheticTerrainTypeEnum(IntEnum):
    HEIGHT_MAP = 0
    HEIGHT_MAP_DISCRETE = 1
    HEIGHT_MAP_STEPS = 2
    HEIGHT_MAP_STAIRS = 3
    HEIGHT_MAP_BINARY = 4
    STANDARD_STAIRS = 5
    OPEN_STAIRS = 6
    LEDGE_STAIRS = 7
    RANDOM_BOXES = 8
