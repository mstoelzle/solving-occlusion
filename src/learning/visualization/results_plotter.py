import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from progress.bar import Bar
import seaborn as sns
import torch
from typing import *
import warnings

from .sample_plotter import draw_error_uncertainty_plot, draw_solutions_plot
from src.enums import *
from src.utils.log import get_logger

logger = get_logger("results_plotter")
sns.set(style="whitegrid")


class ResultsPlotter:
    def __init__(self, results_hdf5_path: pathlib.Path, logdir: pathlib.Path,
                 remote: bool = False, **kwargs):
        self.config = kwargs
        self.logdir = logdir

        self.results_hdf5_path = results_hdf5_path
        self.remote = remote

        self.results_hdf5_file: Optional[h5py.File] = None

    def plot(self):
        self.results_hdf5_file: h5py.File = h5py.File(str(self.results_hdf5_path), 'r')
        self.results_hdf5_file.__enter__()

        with self.results_hdf5_file:
            for task_string, task_hdf5_group in self.results_hdf5_file.items():
                task_string_split = task_string.split("_")
                task_uid = int(task_string_split[1])

                self.plot_task(task_uid, task_hdf5_group)

    def plot_task(self, task_uid: int, task_hdf5_group: h5py.Group):
        logdir = self.logdir / f"task_{task_uid}"

        logger.info(f"Plot task {task_uid}")
        if self.config.get("sample_frequency", 0) > 0:
            for purpose, purpose_hdf5_group in task_hdf5_group.items():
                self.save_samples(purpose_hdf5_group, logdir / f"{purpose}_samples")

        if self.config.get("correlation_occluded_area", False) is True:
            for purpose, purpose_hdf5_group in task_hdf5_group.items():
                if 'loss' in purpose_hdf5_group:
                    self.plot_correlation_area_occluded(purpose_hdf5_group, logdir / f"{purpose}_analysis")

    def save_samples(self, purpose_hdf5_group: h5py.Group, logdir: pathlib.Path):
        logdir.mkdir(exist_ok=True, parents=True)
        data_hdf5_group = purpose_hdf5_group[f"data"]
        dataset_length = len(data_hdf5_group[ChannelEnum.REC_DEM.value])
        num_samples = int(dataset_length / self.config["sample_frequency"])

        progress_bar = Bar(f"Plot samples for {str(purpose_hdf5_group.name)}", max=num_samples)
        for sample_idx in range(num_samples):
            idx = sample_idx * self.config["sample_frequency"]
            res_grid = data_hdf5_group[ChannelEnum.RES_GRID.value][idx, ...]
            rel_position = data_hdf5_group[ChannelEnum.REL_POSITION.value][idx, ...]
            rec_dem = data_hdf5_group[ChannelEnum.REC_DEM.value][idx, ...]
            occluded_elevation_map = data_hdf5_group[ChannelEnum.OCC_DEM.value][idx, ...]
            comp_dem = data_hdf5_group[ChannelEnum.COMP_DEM.value][idx, ...]

            gt_dem = None
            if ChannelEnum.GT_DEM.value in data_hdf5_group:
                gt_dem = data_hdf5_group[ChannelEnum.GT_DEM.value][idx, ...]

            non_occluded_elevation_map = occluded_elevation_map[~np.isnan(occluded_elevation_map)]

            rec_data_um = None
            if ChannelEnum.REC_DATA_UM.value in data_hdf5_group:
                rec_data_um = data_hdf5_group[ChannelEnum.REC_DATA_UM.value][idx, ...]
            comp_data_um = None
            if ChannelEnum.COMP_DATA_UM.value in data_hdf5_group:
                comp_data_um = data_hdf5_group[ChannelEnum.COMP_DATA_UM.value][idx, ...]
            model_um = None
            if ChannelEnum.MODEL_UM.value in data_hdf5_group:
                model_um = data_hdf5_group[ChannelEnum.MODEL_UM.value][idx, ...]
            total_um = None
            if ChannelEnum.TOTAL_UM.value in data_hdf5_group:
                total_um = data_hdf5_group[ChannelEnum.TOTAL_UM.value][idx, ...]

            rec_dems = None
            if ChannelEnum.REC_DEMS.value in data_hdf5_group:
                rec_dems = data_hdf5_group[ChannelEnum.REC_DEMS.value][idx, ...]
            comp_dems = None
            if ChannelEnum.COMP_DEMS.value in data_hdf5_group:
                comp_dems = data_hdf5_group[ChannelEnum.COMP_DEMS.value][idx, ...]

            robot_plot_x = int(occluded_elevation_map.shape[0] / 2 + rel_position[0] / res_grid[0])
            robot_plot_y = int(occluded_elevation_map.shape[1] / 2 + rel_position[1] / res_grid[1])
            # we only visualize the robot position if its inside the elevation map
            plot_robot_position = 0 < robot_plot_x < occluded_elevation_map.shape[0] \
                                  and 0 < robot_plot_y < occluded_elevation_map.shape[1]
            if plot_robot_position:
                robot_plot_position = np.array([robot_plot_x, robot_plot_y])
            else:
                robot_plot_position = None

            # 2D
            elevation_vmin = np.min([np.min(non_occluded_elevation_map), np.min(rec_dem),
                                     np.min(comp_dem)])
            elevation_vmax = np.max([np.max(non_occluded_elevation_map), np.max(rec_dem),
                                     np.max(comp_dem)])
            if gt_dem is not None:
                ground_truth_dem_vmin = np.min(gt_dem[~np.isnan(gt_dem)])
                ground_truth_dem_vmax = np.max(gt_dem[~np.isnan(gt_dem)])
                elevation_vmin = np.min([elevation_vmin, ground_truth_dem_vmin])
                elevation_vmax = np.max([elevation_vmax, ground_truth_dem_vmax])

            elevation_cmap = plt.get_cmap("viridis")

            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[10, 10])
            # axes = np.expand_dims(axes, axis=0)

            if gt_dem is not None:
                axes[0, 0].set_title("Ground-truth")
                # matshow plots x and y swapped
                mat = axes[0, 0].matshow(np.swapaxes(gt_dem, 0, 1), vmin=elevation_vmin,
                                         vmax=elevation_vmax, cmap=elevation_cmap)
            axes[0, 1].set_title("Reconstruction")
            # matshow plots x and y swapped
            mat = axes[0, 1].matshow(np.swapaxes(rec_dem, 0, 1), vmin=elevation_vmin,
                                     vmax=elevation_vmax, cmap=elevation_cmap)
            axes[1, 0].set_title("Composition")
            # matshow plots x and y swapped
            mat = axes[1, 0].matshow(np.swapaxes(comp_dem, 0, 1), vmin=elevation_vmin,
                                     vmax=elevation_vmax, cmap=elevation_cmap)
            axes[1, 1].set_title("Occlusion")
            # matshow plots x and y swapped
            mat = axes[1, 1].matshow(np.swapaxes(occluded_elevation_map, 0, 1), vmin=elevation_vmin,
                                     vmax=elevation_vmax, cmap=elevation_cmap)
            fig.colorbar(mat, ax=axes.ravel().tolist(), fraction=0.045)

            for i, ax in enumerate(axes.reshape(-1)):
                if plot_robot_position:
                    ax.plot([robot_plot_x], [robot_plot_y], marker="*", color="red")

                # Hide grid lines
                ax.grid(False)

            plt.draw()
            plt.savefig(str(logdir / f"sample_2d_{idx}.pdf"))
            if self.remote is not True:
                plt.show()
            plt.close()

            # 3D
            fig = plt.figure(figsize=[2 * 6.4, 1 * 4.8])
            plt.clf()
            axes = []
            num_cols = 3

            x_3d = np.arange(start=-int(occluded_elevation_map.shape[0] / 2),
                             stop=int(occluded_elevation_map.shape[0] / 2)) * res_grid[0]
            y_3d = np.arange(start=-int(occluded_elevation_map.shape[1] / 2),
                             stop=int(occluded_elevation_map.shape[1] / 2)) * res_grid[1]
            x_3d, y_3d = np.meshgrid(x_3d, y_3d)

            axes.append(fig.add_subplot(100 + num_cols * 10 + 1, projection="3d"))
            # the np.NaNs in the occluded elevation maps give us these warnings:
            warnings.filterwarnings("ignore", category=UserWarning)
            if gt_dem is not None:
                axes[0].set_title("Ground-truth")
                axes[0].plot_surface(x_3d, y_3d, gt_dem, vmin=elevation_vmin, vmax=elevation_vmax,
                                     cmap=elevation_cmap)
            axes.append(fig.add_subplot(100 + num_cols * 10 + 2, projection="3d"))
            axes[1].set_title("Reconstruction")
            axes[1].plot_surface(x_3d, y_3d, rec_dem, vmin=elevation_vmin, vmax=elevation_vmax,
                                 cmap=elevation_cmap)
            axes.append(fig.add_subplot(100 + num_cols * 10 + 3, projection="3d"))
            axes[2].set_title("Occlusion")
            axes[2].plot_surface(x_3d, y_3d, occluded_elevation_map, vmin=elevation_vmin, vmax=elevation_vmax,
                                 cmap=elevation_cmap)
            warnings.filterwarnings("default", category=UserWarning)
            fig.colorbar(mat, ax=axes, fraction=0.015)

            for i, ax in enumerate(axes):
                if plot_robot_position:
                    ax.scatter([rel_position[0]], [rel_position[1]], [rel_position[2]], marker="*", color="red")
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")
                ax.set_zlabel("z [m]")

                # Hide grid lines
                ax.grid(False)

            plt.draw()
            plt.savefig(str(logdir / f"sample_3d_{idx}.pdf"))
            if self.remote is not True:
                plt.show()
            plt.close()

            if gt_dem is not None \
                    or rec_data_um is not None or model_um is not None:
                draw_error_uncertainty_plot(idx, logdir,
                                            gt_dem, rec_dem, comp_dem,
                                            rec_data_um=rec_data_um, comp_data_um=comp_data_um,
                                            model_um=model_um, total_um=total_um,
                                            robot_plot_position=robot_plot_position, remote=self.remote,
                                            indiv_vranges=self.config.get("indiv_vranges", False))

            if rec_dems is not None:
                draw_solutions_plot(idx, logdir, ChannelEnum.REC_DEMS, rec_dems,
                                    robot_plot_position=robot_plot_position, remote=self.remote)

            if comp_dems is not None:
                draw_solutions_plot(idx, logdir, ChannelEnum.COMP_DEMS, rec_dems,
                                    robot_plot_position=robot_plot_position, remote=self.remote)

            progress_bar.next()
        progress_bar.finish()

    def plot_correlation_area_occluded(self, purpose_hdf5_group: h5py.Group, logdir: pathlib.Path):
        logdir.mkdir(exist_ok=True, parents=True)
        loss_hdf5_group = purpose_hdf5_group["loss"]

        # percentage of entire area which is occluded
        occ_mask = purpose_hdf5_group["data"][ChannelEnum.OCC_MASK.value]
        shape_map = occ_mask.shape
        occluded_area = np.sum(occ_mask, axis=(1, 2)) / (shape_map[1] * shape_map[2])

        bins = [0.01, 0.05, 0.10, 0.2, 1]
        box_plot_y = np.zeros(shape=occluded_area.shape)
        bin_decs = []
        lower_bound = 0
        idx = 0
        for higher_bound in bins:
            selector = np.logical_and(lower_bound <= occluded_area,
                                      occluded_area <= higher_bound)

            loss_for_bin = np.array(loss_hdf5_group[LossEnum.MSE_REC_OCC.value])
            loss_for_bin = loss_for_bin[selector]
            box_plot_y[idx:idx + loss_for_bin.shape[0]] = loss_for_bin

            bin_description = f"{lower_bound * 100}-{higher_bound * 100}\nN={loss_for_bin.shape[0]}"
            [bin_decs.append(bin_description) for i in range(0, loss_for_bin.shape[0])]

            idx += loss_for_bin.shape[0]
            lower_bound = higher_bound

        df = pd.DataFrame({"area occluded [%]": bin_decs, "reconstruction occlusion loss (MSE)": box_plot_y})

        plt.figure()
        ax = sns.boxplot(x="area occluded [%]", y="reconstruction occlusion loss (MSE)", data=df)
        plt.title("Correlation area-occluded and loss")

        plt.tight_layout()
        plt.draw()
        plt.savefig(str(logdir / f"correlation_area_occluded.pdf"))
        if self.remote is not True:
            plt.show()
        plt.close()
