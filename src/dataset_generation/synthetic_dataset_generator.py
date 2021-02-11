from dataclasses import dataclass
import numpy as np
import os
from pathlib import Path
from progress.bar import Bar
from scipy.spatial.transform import Rotation

from .base_dataset_generator import BaseDatasetGenerator
from src.enums import *


@dataclass
class Position:
    x: float = 0.
    y: float = 0.
    z: float = 0.
    yaw: float = 0.


@dataclass
class ElevationMap:
    terrain_origin_offset: Position  # offset of center of elevation map grid from terrain origin
    gt_dem: np.array
    occ_dem: np.array


@dataclass
class Robot:
    robot_position: Position  # elevation map relative position of robot
    height_viewpoint: float

    elevation_map: ElevationMap


class SyntheticDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # create a generator object
        from height_map_generator import HeightMapGenerator
        self.elevation_map_generator = HeightMapGenerator()
        self.elevation_map_generator.set_seed(self.seed)

        import grid_map_raycasting
        self.grid_map_raycasting = grid_map_raycasting

        # Path to raisim install (e.g. $LOCAL_INSTALL) as saved in the environment variable $RAISIM_INSTALL
        # run this in your terminal: export RAISIM_INSTALL=$LOCAL_INSTALL
        raisim_install_path = Path(os.getenv("RAISIM_INSTALL"))
        # We need to set the path to the raisim license file
        # It is usually placed in the rsc folder of the raisim installation
        grid_map_raycasting.setRaisimLicenseFile(str(raisim_install_path / "rsc" / "activation.raisim"))

        # get the terrain parameter size
        self.terrain_param_sizes = self.elevation_map_generator.get_param_sizes()

        self.terrain_types = []
        assert len(self.config.get("terrain_types", [])) > 0
        for terrain_type_name in self.config.get("terrain_types", []):
            terrain_type = SyntheticTerrainTypeEnum[terrain_type_name]
            self.terrain_types.append(terrain_type)

        self.terrain_width = self.config["elevation_map"]["size"]
        self.terrain_height = self.config["elevation_map"]["size"]
        self.terrain_resolution = self.config["elevation_map"]["resolution"]

        self.robot_position_config = self.config.get("robot_position", {})

        self.reset()

    def reset(self):
        super().reset()
        self.reset_cache()

    def run(self):
        for purpose in ["train", "val", "test"]:
            num_samples = self.config[f"num_{purpose}_samples"]

            self.hdf5_group = self.hdf5_file.create_group(purpose)

            dataset_shape = (0, self.terrain_height, self.terrain_width)
            dataset_maxshape = (num_samples, self.terrain_height, self.terrain_width)

            self.hdf5_group.create_dataset(name=ChannelEnum.GT_DEM.value,
                                           shape=dataset_shape, maxshape=dataset_maxshape)
            self.hdf5_group.create_dataset(name=ChannelEnum.OCC_MASK.value,
                                           shape=dataset_shape, maxshape=dataset_maxshape)

            progress_bar = Bar(f"Generating {purpose} dataset", max=num_samples)
            sample_idx = 0
            while sample_idx < num_samples:
                terrain_idx = np.random.randint(low=0, high=len(self.terrain_types))
                terrain_type = self.terrain_types[terrain_idx]
                terrain_id = terrain_type.value

                # the numpy array which is given to the
                terrain_param = np.random.rand(self.terrain_param_sizes[terrain_id])

                # generate terrain
                self.elevation_map_generator.generate_new_terrain(terrain_id, terrain_param)

                # allocate the memory for the elevation map
                gt_dem = np.zeros((self.terrain_height, self.terrain_width))

                # TODO: maybe we need to sample the center x and center y positions
                # TODO: activate rotation of terrain origin again
                terrain_yaw = np.random.uniform(0, 2 * np.pi)
                # print("terrain_yaw", terrain_yaw)
                terrain_origin_offset = Position(x=0., y=0., z=0., yaw=terrain_yaw)

                # sample the height map from the generated terrains
                self.elevation_map_generator.get_height_map(terrain_origin_offset.x, terrain_origin_offset.y,
                                                            terrain_origin_offset.yaw,
                                                            self.terrain_height, self.terrain_width,
                                                            self.terrain_resolution,
                                                            gt_dem)

                height_viewpoint_config = self.config["elevation_map"]["height_viewpoint"]
                if type(height_viewpoint_config) is float:
                    height_viewpoint = height_viewpoint_config
                elif type(height_viewpoint_config) is dict:
                    height_viewpoint = np.random.uniform(height_viewpoint_config["min"],
                                                         height_viewpoint_config["max"])
                else:
                    raise ValueError

                # TODO: implement yaw view angle later
                robot_yaw = 0

                if self.robot_position_config == "center":
                    robot_x = 0
                    robot_y = 0
                else:
                    robot_x_config = self.robot_position_config.get("x", 0)
                    if type(robot_x_config) == float or type(robot_x_config) == int:
                        robot_x = robot_x_config
                    elif type(robot_x_config) == dict:
                        delta_x = robot_x_config["max"] - robot_x_config["min"]
                        assert delta_x >= 0
                        sign_x = np.random.choice([-1, 1])
                        robot_x = sign_x * (robot_x_config["min"] + np.random.uniform(0, delta_x))
                    else:
                        raise NotImplementedError

                    robot_y_config = self.robot_position_config.get("y", 0)
                    if type(robot_y_config) == float or type(robot_y_config) == int:
                        robot_y = robot_y_config
                    elif type(robot_y_config) == dict:
                        delta_y = robot_y_config["max"] - robot_y_config["min"]
                        assert delta_y >= 0
                        sign_y = np.random.choice([-1, 1])
                        robot_y = sign_y * (robot_y_config["min"] + np.random.uniform(0, delta_y))
                    else:
                        raise NotImplementedError

                robot_position = Position(x=robot_x, y=robot_y, yaw=robot_yaw)
                elevation_map_object = ElevationMap(terrain_origin_offset=terrain_origin_offset,
                                                    gt_dem=gt_dem,
                                                    occ_dem=None)
                robot = Robot(robot_position=robot_position, height_viewpoint=height_viewpoint,
                              elevation_map=elevation_map_object)

                # get elevation of robot position
                u = int(gt_dem.shape[0] / 2 + robot_position.x / self.terrain_resolution)
                v = int(gt_dem.shape[1] / 2 + robot_position.y / self.terrain_resolution)
                robot_position.z = gt_dem[u, v] + robot.height_viewpoint

                res_grid = np.array([self.terrain_resolution, self.terrain_resolution], dtype=np.double)
                rel_position = np.array([robot_position.x, robot_position.y, robot_position.z])
                rel_attitude = Rotation.from_euler("z", robot_position.yaw).as_quat()

                vantage_point = np.array([robot_position.x, robot_position.y, robot_position.z], dtype=np.double)
                occ_mask = self.grid_map_raycasting.rayCastGridMap(vantage_point, gt_dem, res_grid)

                if np.sum(occ_mask) == 0:
                    # we skip the elevation map if we do not find any occlusion
                    continue

                sample_idx += 1

                self.res_grid.append(res_grid)
                self.rel_positions.append(rel_position)
                self.rel_attitudes.append(rel_attitude)
                self.gt_dems.append(gt_dem)
                self.occ_masks.append(occ_mask)
                self.update_dataset_range(gt_dem)

                if self.initialized_datasets is False:
                    self.create_base_datasets(self.hdf5_group, num_samples)

                if sample_idx % self.config.get("save_frequency", 50) == 0 or sample_idx >= num_samples:
                    self.save_cache()

                self.visualize(sample_idx=sample_idx, res_grid=res_grid, rel_position=rel_position,
                               gt_dem=gt_dem, occ_mask=occ_mask)

                progress_bar.next()
            progress_bar.finish()
            self.write_metadata(self.hdf5_group)
            self.reset()

    def save_cache(self):
        self.extend_dataset(self.hdf5_group[ChannelEnum.GT_DEM.value], self.gt_dems)
        self.extend_dataset(self.hdf5_group[ChannelEnum.OCC_MASK.value], self.occ_masks)

        super().save_cache()
