from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import msgpack
import pathlib
from progress.bar import Bar
from scipy.spatial.transform import Rotation
import torch
from typing import *
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

        self.log = None
        self.reset()

    def reset(self):
        super().reset()

    def __enter__(self):
        self.log = msgpack.unpack(open(self.config["msgpack_path"], "rb"))

        super().__enter__()

    def run(self):
        print("Found the following streams in msgpack: ", self.log.keys())
        occ_dem_msgs = self.log["/ga_slam.elevationMapMean"]
        occ_data_um_msgs = self.log["/ga_slam.elevationMapVariance"]
        timestamps = self.log["/ga_slam.elevationMapMean.meta"]["timestamps"]

        self.hdf5_group = self.hdf5_file

        self.total_num_samples = len(occ_dem_msgs)


        prior_occ_dem = None

        progress_bar = Bar(f"Processing msgspack from {self.config['msgpack_path']}",
                           max=self.total_num_samples)
        sample_idx = 0
        for occ_dem_msg, occ_data_um_msg in zip(occ_dem_msgs, occ_data_um_msgs):
            time = occ_dem_msg["time"]
            h, w = occ_dem_msg["height"], occ_dem_msg["width"]

            res_grid = np.array([0.1, 0.1])  # TODO
            rel_position = np.array([0, 0, 0])  # TODO
            rel_attitude = Rotation.from_euler('zyx', [0, 0, 0]).as_quat()

            occ_dem = np.array(occ_dem_msg["data"])
            occ_dem = occ_dem.reshape((-1, int(np.sqrt(occ_dem.shape[0]))), order="F")

            occ_data_um = np.array(occ_data_um_msg["data"])
            occ_data_um = occ_data_um.reshape((-1, int(np.sqrt(occ_data_um.shape[0]))), order="F")

            if np.isnan(occ_dem).all():
                # we skip because the DEM only contains occlusion (NaNs)
                progress_bar.next()
                continue

            if prior_occ_dem is not None:
                # we compute MSE and PSNR between the current occluded dem and the occluded dem from the prior timestamp
                occ_dem_no_nan = np.nan_to_num(occ_dem, copy=True, nan=0.0)
                prior_occ_dem_no_nan = np.nan_to_num(prior_occ_dem, copy=True, nan=0.0)
                mse = mse_loss_fct(input=torch.tensor(occ_dem_no_nan),
                                   target=torch.tensor(prior_occ_dem_no_nan))
                psnr = psnr_from_mse_loss_fct(mse=mse,
                                              data_min=np.min([occ_dem_no_nan, prior_occ_dem_no_nan]).item(),
                                              data_max=np.max([occ_dem_no_nan, prior_occ_dem_no_nan]).item())

                # we want to exclude dems which are too similar
                if psnr > self.config.get("psnr_similarity_threshold", 50):
                    progress_bar.next()
                    continue

            self.res_grid.append(res_grid)
            self.rel_positions.append(rel_position)
            self.rel_attitudes.append(rel_attitude)
            self.occ_dems.append(occ_dem)
            self.occ_data_ums.append(occ_data_um)

            if self.initialized_datasets is False:
                super().create_base_datasets(self.hdf5_group, self.total_num_samples)

                self.hdf5_group.create_dataset(name=ChannelEnum.OCC_DEM.value,
                                               shape=(0, occ_dem.shape[0], occ_dem.shape[1]),
                                               maxshape=(self.total_num_samples, occ_dem.shape[0], occ_dem.shape[1]))
                self.hdf5_group.create_dataset(name=ChannelEnum.OCC_DATA_UM.value,
                                               shape=(0, occ_data_um.shape[0], occ_data_um.shape[1]),
                                               maxshape=(self.total_num_samples, occ_data_um.shape[0],
                                                         occ_data_um.shape[1]))

            if self.sample_idx % self.config.get("save_frequency", 50) == 0:
                self.save_cache()

            self.visualize(sample_idx=sample_idx, res_grid=res_grid, rel_position=rel_position,
                           occ_dem=occ_dem)

            prior_occ_dem = occ_dem
            sample_idx += 1
            progress_bar.next()

        self.save_cache()
        progress_bar.finish()

    def save_cache(self):
        self.extend_dataset(self.hdf5_group[ChannelEnum.OCC_DEM.value], self.occ_dems)
        # self.extend_dataset(self.hdf5_group[ChannelEnum.GT_DEM.value], self.gt_dems)
        # self.extend_dataset(self.hdf5_group[ChannelEnum.OCC_MASK.value], self.occ_masks)
        self.extend_dataset(self.hdf5_group[ChannelEnum.OCC_DATA_UM.value], self.occ_data_ums)

        super().save_cache()
