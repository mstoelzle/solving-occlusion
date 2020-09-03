from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar

from .base_dataset_generator import BaseDatasetGenerator
from src.enums.terrain_type_enum import TerrainTypeEnum


@dataclass
class RobotPosition:
    x: float = 0.
    y: float = 0.
    yaw: float = 0.


@dataclass
class Robot:
    height_viewpoint: float
    robot_position: RobotPosition
    elevation_map: np.array


class SyntheticDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # create a generator object
        from height_map_generator import HeightMapGenerator
        self.elevation_map_generator = HeightMapGenerator()
        # get the terrain parameter size
        self.terrain_param_sizes = self.elevation_map_generator.get_param_sizes()

        self.terrain_types = []
        assert len(self.config.get("terrain_types", [])) > 0
        for terrain_type_name in self.config.get("terrain_types", []):
            terrain_type = TerrainTypeEnum[terrain_type_name]
            self.terrain_types.append(terrain_type)

        self.terrain_width = self.config["elevation_map"]["size"]
        self.terrain_height = self.config["elevation_map"]["size"]
        self.terrain_resolution = self.config["elevation_map"]["resolution"]

    def run(self):
        for purpose in ["train", "val", "test"]:
            num_samples = self.config[f"num_{purpose}_samples"]
            progress_bar = Bar(f"Generating {purpose} dataset", max=num_samples)
            for sample_idx in range(num_samples):
                terrain_idx = np.random.randint(low=0, high=len(self.terrain_types))
                terrain_type = self.terrain_types[terrain_idx]
                terrain_id = terrain_type.value

                # the numpy array which is given to the
                terrain_param = np.random.rand(self.terrain_param_sizes[terrain_id])

                # generate terrain
                self.elevation_map_generator.generate_new_terrain(terrain_id, terrain_param)

                # allocate the memory for the elevation map
                elevation_map = np.zeros((self.terrain_width, self.terrain_height))

                # you can select the center position and yaw angle of the height map
                yaw_angle = np.random.uniform(0, 2 * np.pi)
                # TODO: maybe we need to sample the center x and center y positions
                robot_position = RobotPosition(0., 0., yaw_angle)

                # sample the height map from the generated terrains
                self.elevation_map_generator.get_height_map(robot_position.x, robot_position.y, robot_position.yaw,
                                                            self.terrain_height, self.terrain_width,
                                                            self.terrain_resolution,
                                                            elevation_map)

                height_viewpoint_config = self.config["elevation_map"]["height_viewpoint"]
                if type(height_viewpoint_config) is float:
                    height_viewpoint = height_viewpoint_config
                elif type(height_viewpoint_config) is dict:
                    height_viewpoint = np.random.uniform(0.3, 0.8)
                else:
                    raise ValueError

                robot = Robot(height_viewpoint, robot_position, elevation_map)

                occluded_elevation_map = self.trace_occlusion(robot)

                if self.config.get("visualization", None) is not None:
                    if self.config["visualization"] is True \
                            or sample_idx % self.config["visualization"].get("frequency", 100) == 0:
                        fig, axes = plt.subplots(nrows=1, ncols=2)
                        mat = axes[0].matshow(elevation_map)
                        axes[0].plot([self.terrain_width / 2], [self.terrain_height / 2], marker="*", color="red")
                        mat = axes[1].matshow(occluded_elevation_map)
                        axes[1].plot([self.terrain_width / 2], [self.terrain_height / 2], marker="*", color="red")

                        fig.colorbar(mat, ax=axes.ravel().tolist(), fraction=0.021)
                        plt.show()

                progress_bar.next()
            progress_bar.finish()

    def trace_occlusion(self, robot: Robot) -> np.array:
        """
        Trace occlusion starting from (0,0) to each cell
        If the ray to a cell is occluded, we insert a NaN for that cell
        """

        elevation_map = robot.elevation_map
        occluded_height_map = np.copy(elevation_map)

        robot_i = int(elevation_map.shape[0] / 2)
        robot_j = int(elevation_map.shape[1] / 2)

        for i in range(elevation_map.shape[0]):
            for j in range(elevation_map.shape[1]):
                # relative coordinates from position of rover
                relative_x = (i - robot_i) * self.terrain_resolution
                relative_y = (j - robot_j) * self.terrain_resolution
                relative_z = elevation_map[i][j] - (robot.height_viewpoint + elevation_map[robot_i][robot_j])

                # print("position", relative_x, relative_y, relative_z)

                # roll angle to target cell
                theta = np.arctan2(relative_z, np.sqrt(relative_x ** 2 + relative_y ** 2))
                # yaw angle to target cell
                psi = np.arctan2(relative_y, relative_x)

                # print("psi", psi / np.pi * 180)

                ray_x = np.arange(start=0, stop=np.sqrt(relative_x ** 2 + relative_y ** 2),
                                  step=self.terrain_resolution)

                ray_z = robot.height_viewpoint + elevation_map[robot_i][robot_j] + np.tan(theta) * ray_x
                ray = np.stack((ray_x, ray_z), axis=1)

                ray_trace_input_1d = np.zeros([ray_x.shape[0], 2])
                ray_trace_input_1d[:, 0] = ray_x
                ray_trace_world = self.robot_to_world_coordinates(ray_trace_input_1d, robot.robot_position,
                                                                  additional_yaw=-psi)

                ray_trace_world_scan = np.zeros([ray_x.shape[0], 3])
                ray_trace_world_scan[:, 0:2] = ray_trace_world
                scan = np.zeros([ray_x.shape[0]])
                self.elevation_map_generator.get_height_scan(ray_trace_world, scan)
                ray_trace_world_scan[:, 2] = scan

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(ray_trace_world[:, 0], ray_trace_world[:, 1], ray_trace_world_scan[:, 2])
                # ax.scatter(ray_trace_world[:, 0], ray_trace_world[:, 1], ray[:, 1])
                # plt.show()

                occlusion_condition = ray_trace_world_scan[:, 2] <= ray[:, 1]

                if np.sum(occlusion_condition) < ray_trace_world_scan.shape[0]:
                    occluded_height_map[i][j] = np.NaN

        return occluded_height_map

    @staticmethod
    def robot_to_world_coordinates(body_coordinates: np.array, robot_position: RobotPosition,
                                   additional_yaw: float) -> np.array:
        # robot_coordinates: Nx2 array
        world_coordinates = np.copy(body_coordinates)

        yaw = robot_position.yaw + additional_yaw
        world_coordinates[:, 0] = (body_coordinates[:, 0] - robot_position.x) * np.cos(yaw) \
                                  + (body_coordinates[:, 1] - robot_position.y) * np.sin(yaw)
        world_coordinates[:, 1] = -(body_coordinates[:, 0] - robot_position.x) * np.sin(yaw) \
                                  + (body_coordinates[:, 1] - robot_position.y) * np.cos(yaw)

        return world_coordinates
