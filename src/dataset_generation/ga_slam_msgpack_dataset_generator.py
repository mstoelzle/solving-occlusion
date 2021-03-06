from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import msgpack
import pathlib
from progress.bar import Bar
from scipy.spatial.transform import Rotation
import torch
from typing import *
import os
import warnings

from .base_dataset_generator import BaseDatasetGenerator
from src.enums import *
from src.learning.loss.loss import psnr_from_mse_loss_fct, mse_loss_fct


class GASlamMsgpackDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.split_config = self.config.get("split")
        assert list(self.split_config.keys()) == ["train", "val", "test"]
        self.split_msg_indices = {}

        self.unpacker = None
        self.reset()

    def reset(self):
        super().reset()

    def __enter__(self):
        super().__enter__()

        self.msgpack_file = open(self.config["msgpack_path"], 'rb')
        self.unpacker = msgpack.Unpacker(self.msgpack_file, raw=False, max_buffer_size=2 ** 16 - 1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

        self.msgpack_file.close()

    def run(self):
        self.hdf5_group = self.hdf5_file  # TODO: split into train, val and test set

        prior_occ_dem = None
        progress_bar = None
        sample_idx = 0

        self.logger.info("Start loading first chunk of msgpack")

        for chunk_idx, chunk in enumerate(self.unpacker):
            self.logger.info(f"Msgpack chunk {chunk_idx} is loaded")

            occ_dem_msgs = chunk["/ga_slam.localElevationMapMean"]
            occ_data_um_msgs = chunk["/ga_slam.localElevationMapVariance"]
            gt_dem_msgs = chunk["/ga_slam.globalElevationMapMean"]
            gt_data_um_msgs = chunk["/ga_slam.globalElevationMapVariance"]

            for msg in zip(occ_dem_msgs, occ_data_um_msgs, gt_dem_msgs, gt_data_um_msgs):
                occ_dem_msg, occ_data_um_msg, gt_dem_msg, gt_data_um_msg = msg
                time = occ_dem_msg["time"]
                h, w = occ_dem_msg["height"], occ_dem_msg["width"]

                occ_dem = np.array(occ_dem_msg["data"])
                occ_dem = occ_dem.reshape((-1, int(np.sqrt(occ_dem.shape[0]))), order="F")

                occ_data_um = np.array(occ_data_um_msg["data"])
                occ_data_um = occ_data_um.reshape((-1, int(np.sqrt(occ_data_um.shape[0]))), order="F")

                gt_dem = np.array(gt_dem_msg["data"])
                gt_dem = gt_dem.reshape((-1, int(np.sqrt(gt_dem.shape[0]))), order="F")

                gt_data_um = np.array(gt_data_um_msg["data"])
                gt_data_um = gt_data_um.reshape((-1, int(np.sqrt(gt_data_um.shape[0]))), order="F")

                res_grid = np.array([0.05, 0.05])
                rel_position_z = occ_dem[int(occ_dem.shape[0] // 2), int(occ_dem.shape[1] // 2)]
                rel_position = np.array([0, 0, rel_position_z])
                rel_attitude = Rotation.from_euler('zyx', [0, 0, 0]).as_quat()

                # self.visualize(sample_idx=sample_idx, res_grid=res_grid, rel_position=rel_position,
                #                occ_dem=occ_dem, gt_dem=gt_dem, occ_data_um=occ_data_um, gt_data_um=gt_data_um)

                target_size_x = self.config.get("size", occ_dem.shape[0])
                target_size_y = self.config.get("size", occ_dem.shape[1])
                num_subgrids_x = int(np.floor(occ_dem.shape[0] / target_size_x))
                num_subgrids_y = int(np.floor(occ_dem.shape[1] / target_size_y))

                assert num_subgrids_x >= 1 and num_subgrids_y >= 1

                if progress_bar is None:
                    # we extrapolate the total maximum number of samples
                    # by comparing the number of messages and size of the current chunk
                    # TODO: I am not sure if this code is correct (for multiple chunks)
                    file_size = os.path.getsize(self.config["msgpack_path"])  # in bytes
                    self.total_num_samples = int(len(occ_dem_msgs) / self.unpacker.tell() * file_size)
                    self.total_num_samples *= num_subgrids_x * num_subgrids_y  # multiply with the number of subgrids

                    progress_bar = Bar(f"Processing msgspack from {self.config['msgpack_path']}",
                                       max=self.total_num_samples)

                start_x = 0
                for i in range(num_subgrids_x):
                    stop_x = start_x + target_size_x
                    start_y = 0
                    for j in range(num_subgrids_y):
                        stop_y = start_y + target_size_y

                        occ_dem_subgrid = occ_dem[start_x:stop_x, start_y:stop_y]
                        occ_data_um_subgrid = occ_data_um[start_x:stop_x, start_y:stop_y]
                        gt_dem_subgrid = gt_dem[start_x:stop_x, start_y:stop_y]
                        gt_data_um_subgrid = gt_data_um[start_x:stop_x, start_y:stop_y]

                        subgrid_delta_x = res_grid[0] * (-occ_dem.shape[0] / 2 + start_x + target_size_x / 2)
                        subgrid_delta_y = res_grid[1] * (-occ_dem.shape[1] / 2 + start_y + target_size_y / 2)
                        rel_position_subgrid_z = occ_dem_subgrid[int(target_size_x//2), int(target_size_y//2)]
                        rel_position_subgrid = np.array([rel_position[0] + subgrid_delta_x,
                                                         rel_position[1] + subgrid_delta_y,
                                                         rel_position_subgrid_z])

                        if np.isnan(occ_dem_subgrid).all():
                            # we skip because the DEM only contains occlusion (NaNs)
                            start_y = stop_y
                            progress_bar.next()
                            continue

                        if np.isnan(gt_dem_subgrid).all():
                            # we skip because the DEM only contains missing values (NaNs)
                            pass
                            # start_y = stop_y
                            # progress_bar.next()
                            # continue

                        max_occ_ratio_thresh = self.config.get("max_occlusion_ratio_threshold", 0.5)
                        # we do not want to include the subgrid in the dataset if its occluded to more than 50%
                        if np.isnan(occ_dem_subgrid).sum() > (target_size_x * target_size_y * max_occ_ratio_thresh):
                            start_y = stop_y
                            progress_bar.next()
                            continue

                        if prior_occ_dem is not None:
                            # we compute MSE and PSNR between the current occluded dem and the occluded dem from the prior timestamp
                            prior_occ_dem_subgrid = prior_occ_dem[start_x:stop_x, start_y:stop_y]

                            occ_dem_subgrid_no_nan = np.nan_to_num(occ_dem_subgrid, copy=True, nan=0.0)
                            prior_occ_dem_subgrid_no_nan = np.nan_to_num(prior_occ_dem_subgrid, copy=True, nan=0.0)

                            mse = mse_loss_fct(input=torch.tensor(occ_dem_subgrid_no_nan),
                                               target=torch.tensor(prior_occ_dem_subgrid_no_nan))

                            data_min = np.min([occ_dem_subgrid_no_nan, prior_occ_dem_subgrid_no_nan]).item()
                            data_max = np.max([occ_dem_subgrid_no_nan, prior_occ_dem_subgrid_no_nan]).item()
                            psnr = psnr_from_mse_loss_fct(mse=mse,
                                                          data_min=data_min,
                                                          data_max=data_max)

                            # we want to exclude dems which are too similar
                            if psnr > self.config.get("psnr_similarity_threshold", 50):
                                start_y = stop_y
                                progress_bar.next()
                                continue

                        self.res_grid.append(res_grid)
                        self.rel_positions.append(rel_position_subgrid)
                        self.rel_attitudes.append(rel_attitude)
                        self.occ_dems.append(occ_dem_subgrid)
                        self.occ_data_ums.append(occ_data_um_subgrid)
                        self.gt_dems.append(gt_dem_subgrid)

                        if self.initialized_datasets is False:
                            super().create_base_datasets(self.hdf5_group, self.total_num_samples)

                            self.hdf5_group.create_dataset(name=ChannelEnum.OCC_DEM.value,
                                                           shape=(0, occ_dem_subgrid.shape[0], occ_dem_subgrid.shape[1]),
                                                           maxshape=(self.total_num_samples,
                                                           occ_dem_subgrid.shape[0], occ_dem_subgrid.shape[1]))
                            self.hdf5_group.create_dataset(name=ChannelEnum.OCC_DATA_UM.value,
                                                           shape=(0, occ_data_um_subgrid.shape[0],
                                                                  occ_data_um_subgrid.shape[1]),
                                                           maxshape=(self.total_num_samples,
                                                                     occ_data_um_subgrid.shape[0],
                                                                     occ_data_um_subgrid.shape[1]))
                            self.hdf5_group.create_dataset(name=ChannelEnum.GT_DEM.value,
                                                           shape=(0, gt_dem_subgrid.shape[0], gt_dem_subgrid.shape[1]),
                                                           maxshape=(self.total_num_samples,
                                                                     gt_dem_subgrid.shape[0], gt_dem_subgrid.shape[1]))

                        if self.sample_idx % self.config.get("save_frequency", 50) == 0:
                            self.save_cache()

                        self.visualize(sample_idx=sample_idx, res_grid=res_grid, rel_position=rel_position_subgrid,
                                       occ_dem=occ_dem_subgrid, gt_dem=gt_dem_subgrid,
                                       occ_data_um=occ_data_um_subgrid, gt_data_um=gt_data_um_subgrid)

                        prior_occ_dem = occ_dem
                        sample_idx += 1
                        start_y = stop_y
                        progress_bar.next()

                    start_x = stop_x

        self.save_cache()
        progress_bar.finish()

    def save_cache(self):
        self.extend_dataset(self.hdf5_group[ChannelEnum.OCC_DEM.value], self.occ_dems)
        self.extend_dataset(self.hdf5_group[ChannelEnum.GT_DEM.value], self.gt_dems)
        self.extend_dataset(self.hdf5_group[ChannelEnum.OCC_DATA_UM.value], self.occ_data_ums)

        super().save_cache()
