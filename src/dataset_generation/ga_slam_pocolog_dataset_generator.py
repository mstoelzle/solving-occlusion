import numpy as np
import torch
from progress.bar import Bar
from scipy.spatial.transform import Rotation

from src.enums import *
from src.learning.loss.loss import psnr_from_mse_loss_fct, mse_loss_fct
from .base_dataset_generator import BaseDatasetGenerator


class GASlamPocologDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reset()

        import pocolog_pybind
        self.pocolog_pybind = pocolog_pybind

    def reset(self):
        super().reset()

        self.hdf5_group = None
        self.multi_file_index = []
        self.streams = []
        self.num_messages = 0
        self.pocolog_num_messages = []

    def run(self):
        pocolog_paths = self.config["pocolog_paths"]
        for purpose in pocolog_paths.keys():
            self.purpose = purpose

            self.initialized_datasets = False
            self.reset_metadata()
            self.hdf5_group = self.hdf5_file.create_group(purpose)

            self.streams = []
            self.multi_file_index = []
            self.num_messages = 0
            self.pocolog_num_messages = []
            for pocolog_idx, pocolog_path in enumerate(pocolog_paths[purpose]):
                multi_file_index = self.pocolog_pybind.pocolog.MultiFileIndex()
                multi_file_index.create_index([pocolog_path])
                self.multi_file_index.append(multi_file_index)

                pocolog_streams = multi_file_index.get_all_streams()
                self.streams.append(pocolog_streams)

                pocolog_num_messages = float('inf')
                for key, stream in pocolog_streams.items():
                    if stream.get_name() in ["/ga_slam.localElevationMapMean",
                                             "/ga_slam.localElevationMapVariance",
                                             "/ga_slam.globalElevationMapMean"]:
                        pocolog_num_messages = min(pocolog_num_messages, stream.get_size())
                self.pocolog_num_messages.append(pocolog_num_messages)
                self.num_messages += pocolog_num_messages

            sample_idx = 0
            for pocolog_idx, pocolog_path in enumerate(pocolog_paths[purpose]):
                print(f"Processing {pocolog_path} for purpose {purpose} "
                      f"with length {self.pocolog_num_messages[pocolog_idx]}")

                occ_dem_stream = self.streams[pocolog_idx]["/ga_slam.localElevationMapMean"]
                occ_data_um_stream = self.streams[pocolog_idx]["/ga_slam.localElevationMapVariance"]
                gt_dem_stream = self.streams[pocolog_idx]["/ga_slam.globalElevationMapMean"]
                gt_data_um_stream = self.streams[pocolog_idx]["/ga_slam.globalElevationMapVariance"]

                prior_occ_dem = None
                progress_bar = None

                for msg_idx in range(self.pocolog_num_messages[pocolog_idx]):
                    occ_dem_compound = occ_dem_stream.get_sample(msg_idx)
                    occ_dem_dict = occ_dem_compound.cast(recursive=True)
                    occ_dem = np.array(occ_dem_dict["data"])
                    occ_dem = occ_dem.reshape((occ_dem_dict["height"], occ_dem_dict["width"]), order="F")
                    occ_dem_compound.destroy()  # we need to clean-up the trace of the Typelib::Value in the heap

                    occ_data_um_compound = occ_data_um_stream.get_sample(msg_idx)
                    occ_data_um_dict = occ_data_um_compound.cast(recursive=True)
                    occ_data_um = np.array(occ_data_um_dict["data"])
                    occ_data_um_newshape = (occ_data_um_dict["height"], occ_data_um_dict["width"])
                    occ_data_um = occ_data_um.reshape(occ_data_um_newshape, order="F")
                    occ_data_um_compound.destroy()  # we need to clean-up the trace of the Typelib::Value in the heap

                    gt_dem_compound = gt_dem_stream.get_sample(msg_idx)
                    gt_dem_dict = gt_dem_compound.cast(recursive=True)
                    gt_dem = np.array(gt_dem_dict["data"])
                    gt_dem = gt_dem.reshape((gt_dem_dict["height"], gt_dem_dict["width"]), order="F")
                    gt_dem_compound.destroy()  # we need to clean-up the trace of the Typelib::Value in the heap

                    gt_data_um_compound = gt_data_um_stream.get_sample(msg_idx)
                    gt_data_um_dict = gt_data_um_compound.cast(recursive=True)
                    gt_data_um = np.array(gt_data_um_dict["data"])
                    gt_data_um = gt_data_um.reshape((gt_data_um_dict["height"], gt_data_um_dict["width"]), order="F")
                    gt_data_um_compound.destroy()  # we need to clean-up the trace of the Typelib::Value in the heap

                    res_grid = np.array([0.05, 0.05])
                    rel_position_z = occ_dem[int(occ_dem.shape[0] // 2), int(occ_dem.shape[1] // 2)]
                    rel_position = np.array([0, 0, rel_position_z])
                    rel_attitude = Rotation.from_euler('zyx', [0, 0, 0]).as_quat()

                    h, w = occ_dem.shape
                    # self.visualize(sample_idx=sample_idx, res_grid=res_grid, rel_position=rel_position,
                    #                occ_dem=occ_dem, gt_dem=gt_dem, occ_data_um=occ_data_um, gt_data_um=gt_data_um)

                    target_size_x = self.config.get("size", occ_dem.shape[0])
                    target_size_y = self.config.get("size", occ_dem.shape[1])
                    num_subgrids_x = int(np.floor(occ_dem.shape[0] / target_size_x))
                    num_subgrids_y = int(np.floor(occ_dem.shape[1] / target_size_y))

                    assert num_subgrids_x >= 1 and num_subgrids_y >= 1

                    if progress_bar is None:
                        # multiply with the number of subgrids
                        self.total_num_samples = self.num_messages * num_subgrids_x * num_subgrids_y

                        progress_bar = Bar(f"Processing pocolog {pocolog_idx} for purpose {purpose}",
                                           max=self.pocolog_num_messages[pocolog_idx])

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
                            rel_position_subgrid_z = occ_dem_subgrid[int(target_size_x // 2), int(target_size_y // 2)]
                            rel_position_subgrid = np.array([rel_position[0] + subgrid_delta_x,
                                                             rel_position[1] + subgrid_delta_y,
                                                             rel_position_subgrid_z])

                            if np.isnan(occ_dem_subgrid).all():
                                # we skip because the DEM only contains occlusion (NaNs)
                                start_y = stop_y
                                progress_bar.next()
                                continue

                            if np.isnan(occ_dem_subgrid).sum() > (target_size_x * target_size_y / 2):
                                # we do not want to include the subgrid in the dataset if its occluded to more than 50%
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

                            self.update_dataset_range(occ_dem_subgrid)
                            self.update_dataset_range(gt_dem_subgrid)

                            if self.initialized_datasets is False:
                                super().create_base_datasets(self.hdf5_group, self.total_num_samples)

                                self.hdf5_group.create_dataset(name=ChannelEnum.OCC_DEM.value,
                                                               shape=(
                                                               0, occ_dem_subgrid.shape[0], occ_dem_subgrid.shape[1]),
                                                               maxshape=(self.total_num_samples,
                                                                         occ_dem_subgrid.shape[0],
                                                                         occ_dem_subgrid.shape[1]))
                                self.hdf5_group.create_dataset(name=ChannelEnum.OCC_DATA_UM.value,
                                                               shape=(0, occ_data_um_subgrid.shape[0],
                                                                      occ_data_um_subgrid.shape[1]),
                                                               maxshape=(self.total_num_samples,
                                                                         occ_data_um_subgrid.shape[0],
                                                                         occ_data_um_subgrid.shape[1]))
                                gt_shape = (0, gt_dem_subgrid.shape[0], gt_dem_subgrid.shape[1])
                                gt_maxshape = (self.total_num_samples, gt_dem_subgrid.shape[0],
                                               gt_dem_subgrid.shape[1])
                                self.hdf5_group.create_dataset(name=ChannelEnum.GT_DEM.value,
                                                               shape=gt_shape, maxshape=gt_maxshape)

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

            self.write_metadata(self.hdf5_group)

    def save_cache(self):
        self.extend_dataset(self.hdf5_group[ChannelEnum.OCC_DEM.value], self.occ_dems)
        self.extend_dataset(self.hdf5_group[ChannelEnum.GT_DEM.value], self.gt_dems)
        self.extend_dataset(self.hdf5_group[ChannelEnum.OCC_DATA_UM.value], self.occ_data_ums)

        super().save_cache()
