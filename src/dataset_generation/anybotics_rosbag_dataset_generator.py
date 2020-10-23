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
        self.reset_cache()

        self.initialized_datasets = False
        self.hdf5_group = None
        self.params_dataset = None
        self.occluded_elevation_map_dataset = None

        self.sample_idx = 0
        self.max_num_samples = None

    def reset_cache(self):
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

        self.rosbag_topics = ['/elevation_mapping/elevation_map_recordable']

        self.hdf5_group = self.hdf5_file.create_group(purpose)

        progress_bar = None
        for msg_idx, (topic, msg, t) in enumerate(self.bag.read_messages(topics=self.rosbag_topics)):
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

                target_size_x = self.config.get("size", grid_map.shape[0])
                target_size_y = self.config.get("size", grid_map.shape[1])
                num_subgrids_x = int(np.floor(grid_map.shape[0] / target_size_x))
                num_subgrids_y = int(np.floor(grid_map.shape[0] / target_size_y))

                if self.max_num_samples is None:
                    self.max_num_samples = self.bag.get_message_count(topic_filters=self.rosbag_topics)
                    self.max_num_samples *= num_subgrids_x * num_subgrids_y
                    progress_bar = Bar(f"Reading anybotics bag", max=self.max_num_samples)

                start_x = 0
                start_y = 0
                for i in range(num_subgrids_x):
                    stop_x = start_x + target_size_x
                    start_y = 0
                    for j in range(num_subgrids_y):
                        stop_y = start_y + target_size_y

                        subgrid = grid_map[start_x:stop_x, start_y:stop_y]

                        if np.isnan(subgrid).sum() > (target_size_x * target_size_y / 2):
                            # we do not want to include the subgrid in the dataset if its occluded to more than 50%
                            progress_bar.next()
                            start_y = stop_y
                            continue

                        subgrid_delta_x = resolution * (-grid_map.shape[0]/2 + start_x + target_size_x / 2)
                        subgrid_delta_y = resolution * (-grid_map.shape[1]/2 + start_y + target_size_y / 2)
                        params = np.array([resolution, position[0] + subgrid_delta_x, position[1] + subgrid_delta_y,
                                           position[2], yaw])

                        self.process_item(params, subgrid)

                        self.sample_idx += 1
                        progress_bar.next()
                        start_y = stop_y

                    start_x = stop_x

        self.extend_dataset(self.params_dataset, self.params)
        self.extend_dataset(self.occluded_elevation_map_dataset, self.occluded_elevation_maps)
        self.reset()

        progress_bar.finish()

    def process_item(self, params: np.array, subgrid: np.array):
        self.params.append(params)
        self.occluded_elevation_maps.append(subgrid)

        if self.initialized_datasets is False:
            dataset_shape = (0, subgrid.shape[0], subgrid.shape[1])
            dataset_maxshape = (self.max_num_samples, subgrid.shape[0], subgrid.shape[1])
            self.params_dataset = self.hdf5_group.create_dataset(name=ChannelEnum.PARAMS.value,
                                                                 shape=(0, params.shape[0]),
                                                                 maxshape=(self.max_num_samples, params.shape[0]))
            self.occluded_elevation_map_dataset = \
                self.hdf5_group.create_dataset(name=ChannelEnum.OCCLUDED_ELEVATION_MAP.value,
                                               shape=dataset_shape, maxshape=dataset_maxshape)
            self.initialized_datasets = True

        if self.sample_idx % self.config.get("save_frequency", 50) == 0 or \
                self.sample_idx >= self.max_num_samples:
            self.extend_dataset(self.params_dataset, self.params)
            self.extend_dataset(self.occluded_elevation_map_dataset, self.occluded_elevation_maps)
            self.reset_cache()
