from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar

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
    ground_truth_elevation_map: np.array
    occluded_elevation_map: np.array


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

    def reset_cache(self):
        self.params = []
        self.elevation_maps = []
        self.binary_occlusion_maps = []

    def run(self):
        for purpose in ["train", "val", "test"]:
            num_samples = self.config[f"num_{purpose}_samples"]

            hdf5_group = self.hdf5_file.create_group(purpose)

            dataset_shape = (0, self.terrain_height, self.terrain_width)
            dataset_maxshape = (num_samples, self.terrain_height, self.terrain_width)
            params_dim = 5
            params_dataset = hdf5_group.create_dataset(name=ChannelEnum.PARAMS.value, shape=(0, params_dim),
                                                       maxshape=(num_samples, params_dim))
            elevation_map_dataset = hdf5_group.create_dataset(name=ChannelEnum.GROUND_TRUTH_ELEVATION_MAP.value,
                                                              shape=dataset_shape, maxshape=dataset_maxshape)
            binary_occlusion_map_dataset = hdf5_group.create_dataset(name=ChannelEnum.BINARY_OCCLUSION_MAP.value,
                                                                     shape=dataset_shape, maxshape=dataset_maxshape)

            progress_bar = Bar(f"Generating {purpose} dataset", max=num_samples)
            num_accepted_samples = 0
            while num_accepted_samples < num_samples:
                terrain_idx = np.random.randint(low=0, high=len(self.terrain_types))
                terrain_type = self.terrain_types[terrain_idx]
                terrain_id = terrain_type.value

                # the numpy array which is given to the
                terrain_param = np.random.rand(self.terrain_param_sizes[terrain_id])

                # generate terrain
                self.elevation_map_generator.generate_new_terrain(terrain_id, terrain_param)

                # allocate the memory for the elevation map
                elevation_map = np.zeros((self.terrain_height, self.terrain_width))

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
                                                            elevation_map)

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

                # print("robot position rel", robot_x, robot_y)

                robot_position = Position(x=robot_x, y=robot_y, yaw=robot_yaw)
                elevation_map_object = ElevationMap(terrain_origin_offset=terrain_origin_offset,
                                                    ground_truth_elevation_map=elevation_map,
                                                    occluded_elevation_map=None)
                robot = Robot(robot_position=robot_position, height_viewpoint=height_viewpoint,
                              elevation_map=elevation_map_object)

                binary_occlusion_map = self.raycast_occlusion(robot)

                if np.sum(binary_occlusion_map) == 0:
                    # we skip the elevation map if we do not find any occlusion
                    continue

                num_accepted_samples += 1
                self.params.append(np.array([self.terrain_resolution, robot_position.x,
                                             robot_position.y, robot_position.z, robot_position.yaw]))
                self.elevation_maps.append(elevation_map)
                self.binary_occlusion_maps.append(binary_occlusion_map)
                self.update_dataset_range(elevation_map)

                if num_accepted_samples % self.config.get("save_frequency", 50) == 0 or \
                        num_accepted_samples >= num_samples:
                    self.extend_dataset(params_dataset, self.params)
                    self.extend_dataset(elevation_map_dataset, self.elevation_maps)
                    self.extend_dataset(binary_occlusion_map_dataset, self.binary_occlusion_maps)
                    self.reset_cache()

                if self.config.get("visualization", None) is not None:
                    if self.config["visualization"] is True \
                            or num_accepted_samples % self.config["visualization"].get("frequency", 100) == 0:
                        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[2 * 6.4, 1 * 4.8])

                        occluded_elevation_map = elevation_map.copy()
                        occluded_elevation_map[binary_occlusion_map == 1] = np.nan
                        non_occluded_elevation_map = occluded_elevation_map[~np.isnan(occluded_elevation_map)]

                        vmin = np.min([np.min(elevation_map), np.min(non_occluded_elevation_map)])
                        vmax = np.max([np.max(elevation_map), np.max(non_occluded_elevation_map)])
                        cmap = plt.get_cmap("viridis")

                        # matshow plots x and y swapped
                        axes[0].set_title("Ground-truth")
                        mat = axes[0].matshow(np.swapaxes(elevation_map, 0, 1), vmin=vmin, vmax=vmax,
                                              cmap=cmap)
                        # matshow plots x and y swapped
                        axes[1].set_title("Occlusion")
                        mat = axes[1].matshow(np.swapaxes(occluded_elevation_map, 0, 1), vmin=vmin, vmax=vmax,
                                              cmap=cmap)

                        robot_plot_x = self.terrain_height / 2 + robot_position.x / self.terrain_resolution
                        robot_plot_y = self.terrain_width / 2 + robot_position.y / self.terrain_resolution
                        for ax in axes:
                            ax.plot([robot_plot_x], [robot_plot_y], marker="*", color="red")

                            # Hide grid lines
                            ax.grid(False)

                        fig.colorbar(mat, ax=axes.ravel().tolist(), fraction=0.021)

                        sample_dir = self.logdir / f"{purpose}_samples"
                        if not sample_dir.is_dir():
                            sample_dir.mkdir(exist_ok=True, parents=True)

                        plt.draw()
                        plt.savefig(str(sample_dir / f"sample_{num_accepted_samples}.pdf"))
                        if self.remote is False:
                            plt.show()
                        plt.close()

                progress_bar.next()
            progress_bar.finish()
            self.write_metadata(hdf5_group)
            self.reset()

    def raycast_occlusion(self, robot: Robot) -> np.array:
        """
        Trace occlusion starting from (0,0) to each cell
        If the ray to a cell is occluded, we insert a NaN for that cell
        """

        elevation_map = robot.elevation_map.ground_truth_elevation_map
        binary_occlusion_map = np.zeros(shape=elevation_map.shape, dtype=np.bool)

        center_i = int(elevation_map.shape[0] / 2)
        center_j = int(elevation_map.shape[1] / 2)

        robot_relative_position = np.array([[robot.robot_position.x, robot.robot_position.y]])
        robot_world_position = self.body_to_world_coordinates(robot_relative_position,
                                                              body_position=robot.elevation_map.terrain_origin_offset)
        robot_elevation_scan = np.zeros(shape=(1,))
        self.elevation_map_generator.get_height_scan(robot_world_position, robot_elevation_scan)
        robot_planar_elevation = robot_elevation_scan.item()

        camera_elevation = robot_planar_elevation + robot.height_viewpoint
        robot.robot_position.z = camera_elevation

        for i in range(elevation_map.shape[0]):
            for j in range(elevation_map.shape[1]):
                # relative vector from ray-casted pixel to robot camera
                relative_x = (i - center_i) * self.terrain_resolution - robot.robot_position.x
                relative_y = (j - center_j) * self.terrain_resolution - robot.robot_position.y
                relative_z = elevation_map[i][j] - camera_elevation

                # print("relative position", relative_x, relative_y, relative_z)

                # roll angle to target cell
                theta = np.arctan2(relative_z, np.sqrt(relative_x ** 2 + relative_y ** 2))
                # yaw angle to target cell
                psi = np.arctan2(relative_y, relative_x)

                # print("psi", psi / np.pi * 180)

                ray_x = np.arange(start=0, stop=0.99 * np.sqrt(relative_x ** 2 + relative_y ** 2),
                                  step=self.terrain_resolution)

                ray_z = camera_elevation + np.tan(theta) * ray_x
                ray = np.stack((ray_x, ray_z), axis=1)

                ray_cast_input_1d = np.zeros([ray_x.shape[0], 2])
                ray_cast_input_1d[:, 0] = ray_x

                ray_trace_in_elevation_map = self.body_to_world_coordinates(ray_cast_input_1d,
                                                                            Position(x=0, y=0, yaw=-psi))
                ray_trace_in_elevation_map = ray_trace_in_elevation_map + np.array([[robot.robot_position.x,
                                                                                     robot.robot_position.y]])
                ray_trace_world = self.body_to_world_coordinates(ray_trace_in_elevation_map,
                                                                 robot.elevation_map.terrain_origin_offset)

                ray_cast_world_scan = np.zeros([ray_x.shape[0], 3])
                ray_cast_world_scan[:, 0:2] = ray_trace_world
                scan = np.zeros([ray_x.shape[0]])
                self.elevation_map_generator.get_height_scan(ray_trace_world, scan)
                ray_cast_world_scan[:, 2] = scan

                occlusion_condition = ray_cast_world_scan[:, 2] <= ray[:, 1]

                if np.sum(occlusion_condition) < ray_cast_world_scan.shape[0]:
                    binary_occlusion_map[i][j] = 1

                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(ray_trace_world[:, 0], ray_trace_world[:, 1], ray_cast_world_scan[:, 2])
                    # ax.scatter(ray_trace_world[:, 0], ray_trace_world[:, 1], ray[:, 1])
                    # plt.show()

        return binary_occlusion_map

    @staticmethod
    def body_to_world_coordinates(body_coordinates: np.array, body_position: Position,
                                  additional_yaw: float = 0) -> np.array:
        # robot_coordinates: Nx2 array
        world_coordinates = np.copy(body_coordinates)

        yaw = body_position.yaw + additional_yaw
        world_coordinates[:, 0] = (body_coordinates[:, 0] - body_position.x) * np.cos(yaw) \
                                  + (body_coordinates[:, 1] - body_position.y) * np.sin(yaw)
        world_coordinates[:, 1] = -(body_coordinates[:, 0] - body_position.x) * np.sin(yaw) \
                                  + (body_coordinates[:, 1] - body_position.y) * np.cos(yaw)

        return world_coordinates
