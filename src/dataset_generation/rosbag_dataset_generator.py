import collections
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from progress.bar import Bar
from rosbags import rosbag1
from rosbags.serde import ros1_to_cdr, deserialize_cdr
from typing import *
import warnings

from .base_dataset_generator import BaseDatasetGenerator
from src.enums import *
from src.utils.rosbags_utils import register_msg_types


class RosbagDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # initialise ROS msg types with rosbags module
        register_msg_types()

        self.rosbag_paths = self.config.get("rosbag_paths", [])
        self.ros_topics = self.config.get("rosbag_topics")

        self.split_config = self.config.get("split")
        assert list(self.split_config.keys()) == ["train", "val", "test"]
        self.split_msg_indices = {}

        self.bags = []
        self.reset()

    def reset(self):
        super().reset()
        self.reset_cache()

        self.purpose_max_num_samples = {}

    def __enter__(self):
        for rosbag_path in self.rosbag_paths:
            rosbag = rosbag1.Reader(rosbag_path)
            rosbag.__enter__()
            self.bags.append(rosbag)

        super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for rosbag in self.bags:
            rosbag.__exit__(exc_type, exc_val, exc_tb)

        self.bags = []

        super().__exit__(exc_type, exc_val, exc_tb)

    def run(self):
        assert self.ros_topics is not None, "You need to specify the applicable ROS topics in the config"

        num_messages = np.zeros(shape=(len(self.bags), 1))
        for bag_idx, reader in enumerate(self.bags):
            # quick-and-dirty fix to access the number of messages for a list of filter topics in a rosbag
            # https://nelsonslog.wordpress.com/2016/04/06/python3-no-len-for-iterators/

            # source: https://github.com/wbolster/cardinality/blob/master/cardinality.py#L48-L52
            d = collections.deque(enumerate(reader.messages(self.ros_topics), 1), maxlen=1)
            num_messages[bag_idx] = d[0][0] if d else 0

            # this requires a lot of memory, but is pretty fast
            # num_messages[bag_idx] = len(tuple(reader.messages(self.rosbag_topics)))

        total_num_messages = int(num_messages.sum().item())

        msg_idx = -1
        progress_bar = None
        for bag_idx, reader in enumerate(self.bags):
            for topic, msgtype, t, rawdata in reader.messages(self.ros_topics):
                msg = deserialize_cdr(ros1_to_cdr(rawdata, msgtype), msgtype)
                msg_idx += 1

                for layer_idx in range(len(msg.layers)):
                    info = msg.info
                    layer = msg.layers[layer_idx]
                    layer_data = msg.data[layer_idx]

                    length_x = info.length_x
                    length_y = info.length_y
                    resolution = info.resolution
                    res_grid = np.array([resolution, resolution])
                    if resolution == 0.0:
                        warnings.warn("We skip DEMs with resolutions = 0.0")
                        continue

                    pose = info.pose
                    position = np.array([pose.position.x, pose.position.y, pose.position.z])
                    orientation = np.array([pose.orientation.x, pose.orientation.y,
                                            pose.orientation.z, pose.orientation.w])

                    layout = layer_data.layout
                    dims = layout.dim

                    data_array = np.array(layer_data.data)
                    if dims[0].label == "column_index" and dims[1].label == "row_index":
                        # column major
                        size_x = dims[1].size
                        size_y = dims[0].size
                        grid_map = data_array.reshape((size_x, size_y), order='F')
                    elif dims[0].label == "row_index" and dims[1].label == "column_index":
                        # row major
                        size_x = dims[0].size
                        size_y = dims[1].size
                        grid_map = data_array.reshape((size_x, size_y), order='C')
                    else:
                        raise ValueError

                    self.update_dataset_range(grid_map[~np.isnan(grid_map)])

                    target_size_x = self.config.get("size", grid_map.shape[0])
                    target_size_y = self.config.get("size", grid_map.shape[1])
                    num_subgrids_x = int(np.floor(grid_map.shape[0] / target_size_x))
                    num_subgrids_y = int(np.floor(grid_map.shape[1] / target_size_y))

                    assert num_subgrids_x >= 1 and num_subgrids_y >= 1

                    if self.total_num_samples is None:
                        total_split = 0
                        for purpose, purpose_split in self.split_config.items():
                            total_split += purpose_split
                        start_msg_idx = 0
                        for i, (purpose, purpose_split) in enumerate(self.split_config.items()):
                            self.split_msg_indices[purpose] = start_msg_idx
                            purpose_num_msgs = int(purpose_split / total_split * total_num_messages)

                            if i >= len(self.split_config.keys()) - 1:
                                self.purpose_max_num_samples[purpose] = (total_num_messages - start_msg_idx) * \
                                                                        num_subgrids_x * num_subgrids_y + 1
                            else:
                                self.purpose_max_num_samples[purpose] = purpose_num_msgs * \
                                                                        num_subgrids_x * num_subgrids_y + 1
                            start_msg_idx += purpose_num_msgs

                        num_samples = num_messages * num_subgrids_x * num_subgrids_y
                        self.total_num_samples = total_num_messages * num_subgrids_x * num_subgrids_y

                    if progress_bar is None:
                        progress_bar = Bar(f"Reading anymal bag {bag_idx+1} / {len(self.bags)}",
                                           max=num_samples[bag_idx])

                    if self.purpose is None and msg_idx >= self.split_msg_indices["train"]:
                        self.purpose = "train"
                        self.hdf5_group = self.hdf5_file.create_group(self.purpose)
                    elif self.purpose == "train" and msg_idx >= self.split_msg_indices["val"]:
                        self.save_cache()
                        self.write_metadata(self.hdf5_group)
                        self.reset_metadata()
                        self.purpose = "val"
                        self.hdf5_group = self.hdf5_file.create_group(self.purpose)
                        self.initialized_datasets = False
                    elif self.purpose == "val" and msg_idx >= self.split_msg_indices["test"]:
                        self.save_cache()
                        self.write_metadata(self.hdf5_group)
                        self.reset_metadata()
                        self.purpose = "test"
                        self.hdf5_group = self.hdf5_file.create_group(self.purpose)
                        self.initialized_datasets = False
                    else:
                        pass

                    start_x = 0
                    for i in range(num_subgrids_x):
                        stop_x = start_x + target_size_x
                        start_y = 0
                        for j in range(num_subgrids_y):
                            stop_y = start_y + target_size_y

                            subgrid = grid_map[start_x:stop_x, start_y:stop_y]

                            max_occ_ratio_thresh = self.config.get("max_occlusion_ratio_threshold", 0.5)
                            # we do not want to include the subgrid in the dataset if its occluded to more than 50%
                            if np.isnan(subgrid).sum() > (target_size_x * target_size_y * max_occ_ratio_thresh):
                                progress_bar.next()
                                start_y = stop_y
                                continue

                            subgrid_delta_x = resolution * (-grid_map.shape[0]/2 + start_x + target_size_x / 2)
                            subgrid_delta_y = resolution * (-grid_map.shape[1]/2 + start_y + target_size_y / 2)
                            rel_position = np.array([position[0] + subgrid_delta_x,
                                                     position[1] + subgrid_delta_y,
                                                     position[2]])
                            rel_attitude = orientation

                            self.process_item(res_grid, rel_position, rel_attitude, subgrid)

                            self.visualize(sample_idx=self.sample_idx, res_grid=res_grid, rel_position=rel_position,
                                           occ_dem=subgrid)

                            self.sample_idx += 1
                            progress_bar.next()
                            start_y = stop_y

                        start_x = stop_x

                    # we only consider layer_idx=0 for now
                    break

            progress_bar.finish()
            progress_bar = None

        self.save_cache()
        self.write_metadata(self.hdf5_group)

    def process_item(self, res_grid: np.array, rel_position: np.array, rel_attitude: np.array, subgrid: np.array,
                     force_save: bool = False):
        self.res_grid.append(res_grid)
        self.rel_positions.append(rel_position)
        self.rel_attitudes.append(rel_attitude)
        self.occ_dems.append(subgrid)
        self.update_dataset_range(subgrid)

        if self.initialized_datasets is False:
            max_num_samples = self.purpose_max_num_samples[self.purpose]

            super().create_base_datasets(self.hdf5_group, max_num_samples)

            self.hdf5_group.create_dataset(name=ChannelEnum.OCC_DEM.value,
                                           shape=(0, subgrid.shape[0], subgrid.shape[1]),
                                           maxshape=(max_num_samples, subgrid.shape[0], subgrid.shape[1]))

        if self.sample_idx % self.config.get("save_frequency", 50) == 0 or \
                self.sample_idx >= self.total_num_samples or force_save is True:
            self.save_cache()

    def save_cache(self):
        self.extend_dataset(self.hdf5_group[ChannelEnum.OCC_DEM.value], self.occ_dems)

        super().save_cache()
