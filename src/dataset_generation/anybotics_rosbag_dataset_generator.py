from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from progress.bar import Bar
from scipy.spatial.transform import Rotation

from .base_dataset_generator import BaseDatasetGenerator
from src.enums import *


class AnyboticsRosbagDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        import rosbag
        self.rosbag_module = rosbag
        self.rosbag_path = pathlib.Path(self.config["rosbag_path"])

        self.bag = None
        self.reset()

    def reset(self):
        self.params = []
        self.occluded_elevation_maps = []

    def __enter__(self):
        self.bag = self.rosbag_module.Bag(str(self.rosbag_path), 'r')

        super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bag.close()

        super().__exit__(exc_type, exc_val, exc_tb)

    def run(self):
        # TODO: maybe we need to add additional purposes later
        purpose = "test"

        num_samples = self.bag.get_message_count()

        hdf5_group = self.hdf5_file.create_group(purpose)

        params_dataset = None
        occluded_elevation_map_dataset = None

        progress_bar = Bar(f"Reading {purpose} anybotics bag", max=num_samples)

        topics = ['/elevation_mapping/elevation_map_recordable']
        initialized_hdf5_datasets = False
        sample_idx = 0
        for msg_idx, (topic, msg, t) in enumerate(self.bag.read_messages(topics=topics)):
            for layer_idx in range(len(msg.layers)):
                info = msg.info
                layer = msg.layers[layer_idx]
                layer_data = msg.data[layer_idx]

                length_x = info.length_x
                length_y = info.length_y
                resolution = info.resolution
                pose = info.pose
                position = np.array([pose.position.x, pose.position.y, pose.position.z])
                orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
                yaw = Rotation.from_quat(orientation).as_euler('zyx')[2]
                params = np.array([resolution, position[0], position[1], position[2], yaw])

                layout = layer_data.layout
                dims = layout.dim

                array = np.array(layer_data.data)
                if dims[0].label == "column_index" and dims[1].label == "row_index":
                    # column major
                    size_x = dims[1].size
                    size_y = dims[0].size
                    grid_map = array.reshape((size_x, size_y), order='F')
                elif dims[0].label == "row_index" and dims[1].label == "column_index":
                    # row major
                    size_x = dims[0].size
                    size_y = dims[1].size
                    grid_map = array.reshape((size_x, size_y), order='C')
                else:
                    raise ValueError

                self.params.append(params)
                self.occluded_elevation_maps.append(grid_map)

                if initialized_hdf5_datasets is False:
                    dataset_shape = (0, grid_map.shape[0], grid_map.shape[1])
                    dataset_maxshape = (num_samples, grid_map.shape[0], grid_map.shape[1])
                    params_dataset = hdf5_group.create_dataset(name=ChannelEnum.PARAMS.value,
                                                               shape=(0, params.shape[0]),
                                                               maxshape=(num_samples, params.shape[0]))
                    occluded_elevation_map_dataset = hdf5_group.create_dataset(name=ChannelEnum.OCCLUDED_ELEVATION_MAP.value,
                                                                               shape=dataset_shape,
                                                                               maxshape=dataset_maxshape)
                    initialized_hdf5_datasets = True

                if sample_idx % self.config.get("save_frequency", 50) == 0 or \
                        sample_idx >= num_samples:
                    self.extend_dataset(params_dataset, self.params)
                    self.extend_dataset(occluded_elevation_map_dataset, self.occluded_elevation_maps)
                    self.reset()

                sample_idx += 1

            progress_bar.next()
        progress_bar.finish()
