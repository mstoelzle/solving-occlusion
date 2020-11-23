import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import torch
from typing import *
import warnings

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
        dataset_length = len(data_hdf5_group[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP.value])
        num_samples = int(dataset_length / self.config["sample_frequency"])
        for sample_idx in range(num_samples):
            idx = sample_idx * self.config["sample_frequency"]
            params = data_hdf5_group[ChannelEnum.PARAMS.value][idx, ...]
            reconstructed_elevation_map = data_hdf5_group[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP.value][idx, ...]
            occluded_elevation_map = data_hdf5_group[ChannelEnum.OCCLUDED_ELEVATION_MAP.value][idx, ...]
            inpainted_elevation_map = data_hdf5_group[ChannelEnum.COMPOSED_ELEVATION_MAP.value][idx, ...]

            ground_truth_elevation_map = None
            if ChannelEnum.GROUND_TRUTH_ELEVATION_MAP.value in data_hdf5_group:
                ground_truth_elevation_map = data_hdf5_group[ChannelEnum.GROUND_TRUTH_ELEVATION_MAP.value][idx, ...]

            non_occluded_elevation_map = occluded_elevation_map[~np.isnan(occluded_elevation_map)]

            # 2D
            vmin = np.min([np.min(non_occluded_elevation_map), np.min(reconstructed_elevation_map),
                           np.min(inpainted_elevation_map)])
            vmax = np.max([np.max(non_occluded_elevation_map), np.max(reconstructed_elevation_map),
                           np.max(inpainted_elevation_map)])
            if ground_truth_elevation_map is not None:
                vmin = np.min([vmin, np.min(ground_truth_elevation_map)])
                vmax = np.max([vmax, np.max(ground_truth_elevation_map)])

            cmap = plt.get_cmap("viridis")

            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[10, 10])
            # axes = np.expand_dims(axes, axis=0)

            if ground_truth_elevation_map is not None:
                axes[0, 0].set_title("Ground-truth")
                # matshow plots x and y swapped
                mat = axes[0, 0].matshow(np.swapaxes(ground_truth_elevation_map, 0, 1), vmin=vmin, vmax=vmax, cmap=cmap)
            axes[0, 1].set_title("Reconstruction")
            # matshow plots x and y swapped
            mat = axes[0, 1].matshow(np.swapaxes(reconstructed_elevation_map, 0, 1), vmin=vmin, vmax=vmax, cmap=cmap)
            axes[1, 0].set_title("Composition")
            # matshow plots x and y swapped
            mat = axes[1, 0].matshow(np.swapaxes(inpainted_elevation_map, 0, 1), vmin=vmin, vmax=vmax, cmap=cmap)
            axes[1, 1].set_title("Occlusion")
            # matshow plots x and y swapped
            mat = axes[1, 1].matshow(np.swapaxes(occluded_elevation_map, 0, 1), vmin=vmin, vmax=vmax, cmap=cmap)
            fig.colorbar(mat, ax=axes.ravel().tolist(), fraction=0.045)

            terrain_resolution = params[0]
            robot_position_x = params[1]
            robot_position_y = params[2]
            robot_position_z = params[3]
            robot_plot_x = int(occluded_elevation_map.shape[0] / 2 + robot_position_x / terrain_resolution)
            robot_plot_y = int(occluded_elevation_map.shape[1] / 2 + robot_position_y / terrain_resolution)
            # we only visualize the robot position if its inside the elevation map
            plot_robot_position = 0 < robot_plot_x < occluded_elevation_map.shape[0] \
                                    and 0 < robot_plot_y < occluded_elevation_map.shape[1]

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
                             stop=int(occluded_elevation_map.shape[0] / 2)) * terrain_resolution
            y_3d = np.arange(start=-int(occluded_elevation_map.shape[1] / 2),
                             stop=int(occluded_elevation_map.shape[1] / 2)) * terrain_resolution
            x_3d, y_3d = np.meshgrid(x_3d, y_3d)

            axes.append(fig.add_subplot(100 + num_cols * 10 + 1, projection="3d"))
            if ground_truth_elevation_map is not None:
                axes[0].set_title("Ground-truth")
                axes[0].plot_surface(x_3d, y_3d, ground_truth_elevation_map, vmin=vmin, vmax=vmax, cmap=cmap)
            axes.append(fig.add_subplot(100 + num_cols * 10 + 2, projection="3d"))
            axes[1].set_title("Reconstruction")
            axes[1].plot_surface(x_3d, y_3d, reconstructed_elevation_map, vmin=vmin, vmax=vmax, cmap=cmap)
            axes.append(fig.add_subplot(100 + num_cols * 10 + 3, projection="3d"))
            axes[2].set_title("Occlusion")
            # the np.NaNs in the occluded elevation maps give us these warnings:
            warnings.filterwarnings("ignore", category=UserWarning)
            axes[2].plot_surface(x_3d, y_3d, occluded_elevation_map, vmin=vmin, vmax=vmax, cmap=cmap)
            warnings.filterwarnings("default", category=UserWarning)
            fig.colorbar(mat, ax=axes, fraction=0.015)

            for i, ax in enumerate(axes):
                if plot_robot_position:
                    ax.scatter([robot_position_x], [robot_position_y],
                               [robot_position_z], marker="*", color="red")
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

            if ground_truth_elevation_map is not None:
                fig = plt.figure(figsize=[1.75 * 6.4, 1.25 * 4.8])
                plt.clf()
                axes = []

                axes.append(fig.add_subplot(121))
                axes[0].set_title("Reconstruction error")
                abs_error = np.abs(reconstructed_elevation_map - ground_truth_elevation_map)
                # matshow plots x and y swapped
                mat = axes[0].matshow(np.swapaxes(abs_error, 0, 1), cmap=plt.get_cmap("RdYlGn_r"))
                if plot_robot_position:
                    axes[0].plot(robot_plot_x, robot_plot_y, marker="*", color="blue")
                axes[0].grid(False)

                axes.append(fig.add_subplot(122, projection="3d"))
                axes[1].set_title("Reconstruction error")
                axes[1].plot_surface(x_3d, y_3d, np.abs(reconstructed_elevation_map - ground_truth_elevation_map),
                                     cmap=plt.get_cmap("RdYlGn_r"))
                axes[1].set_xlabel("x [m]")
                axes[1].set_ylabel("y [m]")
                axes[1].set_zlabel("z [m]")

                fig.colorbar(mat, ax=axes, fraction=0.021)

                plt.draw()
                plt.savefig(str(logdir / f"reconstruction_error_{idx}.pdf"))
                if self.remote is not True:
                    plt.show()
                plt.close()

    def plot_correlation_area_occluded(self, purpose_hdf5_group: h5py.Group, logdir: pathlib.Path):
        logdir.mkdir(exist_ok=True, parents=True)
        loss_hdf5_group = purpose_hdf5_group["loss"]

        # percentage of entire area which is occluded
        binary_occlusion_map = purpose_hdf5_group["data"][ChannelEnum.BINARY_OCCLUSION_MAP.value]
        shape_map = binary_occlusion_map.shape
        occluded_area = np.sum(binary_occlusion_map, axis=(1, 2)) / (shape_map[1] * shape_map[2])

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
