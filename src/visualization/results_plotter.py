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

from .sample_plotter import draw_error_uncertainty_plot, draw_solutions_plot, draw_traversability_plot, \
    draw_qualitative_comparison_plot, draw_occ_mask_plot
from src.enums import *
from src.learning.loss.loss import masked_loss_fct, mse_loss_fct, l1_loss_fct, psnr_loss_fct
from src.utils.log import get_logger
from src.visualization.live_inference_plotter import plot_live_inference

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

        with self.results_hdf5_file:
            for task_string, task_hdf5_group in self.results_hdf5_file.items():
                task_string_split = task_string.split("_")
                task_uid = int(task_string_split[1])

                self.plot_task(task_uid, task_hdf5_group)

            if (isinstance(self.config.get("loss_magnitude_distribution", False), bool)
                    and self.config.get("loss_magnitude_distribution", False)) \
                or (isinstance(self.config.get("loss_magnitude_distribution", []), list)
                    and len(self.config.get("loss_magnitude_distribution", [])) > 0):
                self.plot_loss_magnitude_distribution()

            if len(self.config.get("qualitative_comparison", [])) > 0:
                self.plot_qualitative_comparison()

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

        if self.config.get("live_inference", False) is True and self.remote is False:
            for purpose, purpose_hdf5_group in task_hdf5_group.items():
                plot_live_inference(purpose_hdf5_group)

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
            occ_dem = data_hdf5_group[ChannelEnum.OCC_DEM.value][idx, ...]
            occ_mask = data_hdf5_group[ChannelEnum.OCC_MASK.value][idx, ...]
            rec_dem = data_hdf5_group[ChannelEnum.REC_DEM.value][idx, ...]
            comp_dem = data_hdf5_group[ChannelEnum.COMP_DEM.value][idx, ...]

            gt_dem = None
            if ChannelEnum.GT_DEM.value in data_hdf5_group:
                gt_dem = data_hdf5_group[ChannelEnum.GT_DEM.value][idx, ...]

            nocc_dem = occ_dem[~np.isnan(occ_dem)]

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

            u = int(round(occ_dem.shape[0] / 2 + rel_position[0] / res_grid[0]))
            v = int(round(occ_dem.shape[1] / 2 + rel_position[1] / res_grid[1]))
            # we only visualize the robot position if its inside the elevation map
            plot_robot_position = 0 < u < occ_dem.shape[0] and 0 < v < occ_dem.shape[1]
            if plot_robot_position:
                robot_position_pixel = np.array([u, v])
            else:
                robot_position_pixel = None
            indiv_vranges = self.config.get("indiv_vranges", True)

            # 2D
            if indiv_vranges is False:
                elevation_vmin = np.min([np.min(rec_dem), np.min(comp_dem[~np.isnan(comp_dem)])])
                elevation_vmax = np.max([np.max(rec_dem), np.max(comp_dem[~np.isnan(comp_dem)])])

                if nocc_dem.size != 0:
                    elevation_vmin = np.min([elevation_vmin, np.min(nocc_dem)])
                    elevation_vmax = np.max([elevation_vmax, np.max(nocc_dem)])

                if gt_dem is not None and np.isnan(gt_dem).all() is False:
                    ground_truth_dem_vmin = np.min(gt_dem[~np.isnan(gt_dem)])
                    ground_truth_dem_vmax = np.max(gt_dem[~np.isnan(gt_dem)])
                    elevation_vmin = np.min([elevation_vmin, ground_truth_dem_vmin])
                    elevation_vmax = np.max([elevation_vmax, ground_truth_dem_vmax])
            else:
                elevation_vmin = None
                elevation_vmax = None

            elevation_cmap = plt.get_cmap("viridis")

            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[12, 10])
            # axes = np.expand_dims(axes, axis=0)

            if gt_dem is not None:
                axes[0, 0].set_title("Ground-truth")
                # matshow plots x and y swapped
                mat = axes[0, 0].matshow(np.swapaxes(gt_dem, 0, 1), vmin=elevation_vmin,
                                         vmax=elevation_vmax, cmap=elevation_cmap)
                if indiv_vranges:
                    fig.colorbar(mat, ax=axes[0, 0], fraction=0.08)

            axes[0, 1].set_title("Reconstruction")
            # matshow plots x and y swapped
            mat = axes[0, 1].matshow(np.swapaxes(rec_dem, 0, 1), vmin=elevation_vmin,
                                     vmax=elevation_vmax, cmap=elevation_cmap)
            if indiv_vranges:
                fig.colorbar(mat, ax=axes[0, 1], fraction=0.08)
            axes[1, 0].set_title("Composition")
            # matshow plots x and y swapped
            mat = axes[1, 0].matshow(np.swapaxes(comp_dem, 0, 1), vmin=elevation_vmin,
                                     vmax=elevation_vmax, cmap=elevation_cmap)
            if indiv_vranges:
                fig.colorbar(mat, ax=axes[1, 0], fraction=0.08)
            axes[1, 1].set_title("Occlusion")
            # matshow plots x and y swapped
            mat = axes[1, 1].matshow(np.swapaxes(occ_dem, 0, 1), vmin=elevation_vmin,
                                     vmax=elevation_vmax, cmap=elevation_cmap)
            if indiv_vranges:
                fig.colorbar(mat, ax=axes[1, 1], fraction=0.08)

            if indiv_vranges is False:
                fig.colorbar(mat, ax=axes.ravel().tolist(), fraction=0.045)

            for i, ax in enumerate(axes.reshape(-1)):
                if plot_robot_position:
                    ax.plot([u], [v], marker="*", color="red")

                # Hide grid lines
                ax.grid(False)

                # hide ticks
                if self.config.get("hide_ticks", False):
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])

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

            x_3d = np.arange(start=-int(occ_dem.shape[0] / 2),
                             stop=int(occ_dem.shape[0] / 2)) * res_grid[0]
            y_3d = np.arange(start=-int(occ_dem.shape[1] / 2),
                             stop=int(occ_dem.shape[1] / 2)) * res_grid[1]
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
            axes[2].plot_surface(x_3d, y_3d, occ_dem, vmin=elevation_vmin, vmax=elevation_vmax,
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

                # hide ticks
                if self.config.get("hide_ticks", False):
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])

            plt.draw()
            plt.savefig(str(logdir / f"sample_3d_{idx}.pdf"))
            if self.remote is not True:
                plt.show()
            plt.close()

            if gt_dem is not None \
                    or rec_data_um is not None or model_um is not None:
                draw_error_uncertainty_plot(idx, logdir,
                                            gt_dem=gt_dem, rec_dem=rec_dem, comp_dem=comp_dem,
                                            rec_data_um=rec_data_um, comp_data_um=comp_data_um,
                                            model_um=model_um, total_um=total_um,
                                            robot_position_pixel=robot_position_pixel, remote=self.remote,
                                            indiv_vranges=indiv_vranges,
                                            hide_ticks=self.config.get("hide_ticks", False))

            if rec_dems is not None:
                draw_solutions_plot(idx, logdir, ChannelEnum.REC_DEMS, rec_dems,
                                    robot_position_pixel=robot_position_pixel, remote=self.remote,
                                    hide_ticks=self.config.get("hide_ticks", False))

            if comp_dems is not None:
                draw_solutions_plot(idx, logdir, ChannelEnum.COMP_DEMS, rec_dems,
                                    robot_position_pixel=robot_position_pixel, remote=self.remote,
                                    hide_ticks=self.config.get("hide_ticks", False))

            if occ_mask is not None:
                draw_occ_mask_plot(idx, logdir, occ_mask=occ_mask,
                                   robot_position_pixel=robot_position_pixel, remote=self.remote,
                                   hide_ticks=self.config.get("hide_ticks", False))

            if ChannelEnum.REC_TRAV_RISK_MAP.value in data_hdf5_group \
                    and ChannelEnum.COMP_TRAV_RISK_MAP.value in data_hdf5_group:
                rec_trav_risk_map = data_hdf5_group[ChannelEnum.REC_TRAV_RISK_MAP.value][idx, ...]
                comp_trav_risk_map = data_hdf5_group[ChannelEnum.COMP_TRAV_RISK_MAP.value][idx, ...]
                draw_traversability_plot(idx, logdir,
                                         gt_dem=gt_dem, rec_dem=rec_dem, comp_dem=comp_dem,
                                         rec_data_um=rec_data_um, comp_data_um=comp_data_um,
                                         model_um=model_um, total_um=total_um,
                                         rec_trav_risk_map=rec_trav_risk_map, comp_trav_risk_map=comp_trav_risk_map,
                                         robot_position_pixel=robot_position_pixel, remote=self.remote,
                                         hide_ticks=self.config.get("hide_ticks", False))

            progress_bar.next()
        progress_bar.finish()

    def plot_correlation_area_occluded(self, purpose_hdf5_group: h5py.Group, logdir: pathlib.Path):
        logdir.mkdir(exist_ok=True, parents=True)
        loss_hdf5_group = purpose_hdf5_group["loss"]

        # percentage of entire area which is occluded
        occ_mask = purpose_hdf5_group["data"][ChannelEnum.OCC_MASK.value]
        shape_map = occ_mask.shape
        occlusion_ratio = np.sum(occ_mask, axis=(1, 2)) / (shape_map[1] * shape_map[2])

        bins = [0.01, 0.05, 0.10, 0.2, 1]
        box_plot_y = np.zeros(shape=occlusion_ratio.shape)
        bin_decs = []
        lower_bound = 0
        idx = 0
        for higher_bound in bins:
            selector = np.logical_and(lower_bound <= occlusion_ratio,
                                      occlusion_ratio <= higher_bound)

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

        plt.figure()
        ax = sns.violinplot(y=occlusion_ratio)
        plt.title("Distribution of occlusion ratio")
        plt.tight_layout()
        plt.draw()
        plt.savefig(str(logdir / f"distribution_occlusion_ratio.pdf"))
        if self.remote is not True:
            plt.show()
        plt.close()

    def plot_loss_magnitude_distribution(self):
        for purpose in ["test"]:
            logger.info(f"Plot loss magnitude distribution for purpose {purpose}")

            purpose_logdir = self.logdir / f"{purpose}_analysis"
            purpose_logdir.mkdir(exist_ok=True, parents=True)

            loss_magnitude_config = self.config["loss_magnitude_distribution"]

            x_axis_label_task_uid = False
            if loss_magnitude_config is True:
                x_axis_label_task_uid = True
                loss_magnitude_config = []
                for task_string, _ in self.results_hdf5_file.items():
                    task_string_split = task_string.split("_")
                    task_uid = int(task_string_split[1])
                    loss_magnitude_config.append({"title": str(task_uid), "task": task_uid})

            pd_occ_dict = {"task_uid": [], "l1_rec_occ": [], "mse_rec_occ": []}
            pd_nocc_dict = {"task_uid": [], "l1_rec_nocc": [], "mse_rec_nocc": []}
            for task_spec in loss_magnitude_config:
                method_title = task_spec["title"]
                task_uid = task_spec["task"]

                data_hdf5_group = self.results_hdf5_file[f"task_{task_uid}/{purpose}/data"]
                loss_hdf5_group = self.results_hdf5_file[f"task_{task_uid}/{purpose}/loss"]

                gt_dem = torch.tensor(np.array(data_hdf5_group[ChannelEnum.GT_DEM.value]))
                occ_dem = torch.tensor(np.array(data_hdf5_group[ChannelEnum.OCC_DEM.value]))
                occ_mask = torch.tensor(np.array(data_hdf5_group[ChannelEnum.OCC_MASK.value]), dtype=torch.bool)
                rec_dem = torch.tensor(np.array(data_hdf5_group[ChannelEnum.REC_DEM.value]))

                l1_rec_occ = masked_loss_fct(l1_loss_fct, rec_dem,
                                             gt_dem, occ_mask, reduction="none")[occ_mask == True]
                l1_rec_nocc = masked_loss_fct(l1_loss_fct, rec_dem,
                                              gt_dem, ~occ_mask, reduction="none")[occ_mask == False]
                mse_rec_occ = masked_loss_fct(mse_loss_fct, rec_dem,
                                              gt_dem, occ_mask, reduction="none")[occ_mask == True]
                mse_rec_nocc = masked_loss_fct(mse_loss_fct, rec_dem,
                                               gt_dem, ~occ_mask, reduction="none")[occ_mask == False]

                task_uid_occ = [str(method_title)] * l1_rec_occ.shape[0]
                pd_occ_dict["task_uid"].extend(task_uid_occ)
                pd_occ_dict["l1_rec_occ"].extend(l1_rec_occ.tolist())
                pd_occ_dict["mse_rec_occ"].extend(mse_rec_occ.tolist())

                task_uid_nocc = [str(method_title)] * l1_rec_nocc.shape[0]
                pd_nocc_dict["task_uid"].extend(task_uid_nocc)
                pd_nocc_dict["l1_rec_nocc"].extend(l1_rec_nocc.tolist())
                pd_nocc_dict["mse_rec_nocc"].extend(mse_rec_nocc.tolist())

            df_occ = pd.DataFrame(data=pd_occ_dict)
            df_nocc = pd.DataFrame(data=pd_nocc_dict)

            for plot_type in ["boxplot", "violinplot"]:
                for area in ["occ", "nocc"]:
                    fig = plt.figure(figsize=[len(self.results_hdf5_file)/4+6.4, 4.8])

                    if area == "occ":
                        df_plot = df_occ
                    else:
                        df_plot = df_nocc

                    if plot_type == "boxplot":
                        sns.boxplot(x="task_uid", y=f"l1_rec_{area}", data=df_plot, showfliers=False)
                    else:
                        sns.violinplot(x="task_uid", y=f"l1_rec_{area}", data=df_plot, showfliers=False, width=1.0)

                        df_plot_90th = df_plot.quantile(q=0.9, axis=0)
                        l1_90th = df_plot_90th[f"l1_rec_{area}"]
                        plt.ylim(top=l1_90th * 1.1)

                    if x_axis_label_task_uid:
                        plt.xlabel("Task")
                    else:
                        plt.xlabel(None)

                    if area == "occ":
                        plt.ylabel("$\ell_{1,\mathrm{rec},\mathrm{occ}}$ [m]")
                    else:
                        plt.ylabel("$\ell_{1,\mathrm{rec},\mathrm{nocc}}$ [m]")

                    plt.tight_layout()
                    plt.draw()
                    plt.savefig(str(purpose_logdir / f"loss_magnitude_dist_{plot_type}_{area}.pdf"))
                    if self.remote is not True:
                        plt.show()
                    plt.close()

    def plot_qualitative_comparison(self):
        qual_comp_config = self.config["qualitative_comparison"]

        for purpose in ["test"]:
            purpose_logdir = self.logdir / f"{purpose}_qualitative_comparison"
            purpose_logdir.mkdir(exist_ok=True, parents=True)

            dataset_length = None
            for task_spec in qual_comp_config:
                method_title = task_spec["title"]
                task_uid = task_spec["task"]

                data_hdf5_group = self.results_hdf5_file[f"task_{task_uid}/{purpose}/data"]
                task_dataset_length = len(data_hdf5_group[ChannelEnum.REC_DEM.value])

                if dataset_length is None:
                    dataset_length = task_dataset_length
                else:
                    assert dataset_length == task_dataset_length

            num_samples = int(dataset_length / self.config["sample_frequency"])

            progress_bar = Bar(f"Plot qualitative comparison for {purpose}", max=num_samples)
            for sample_idx in range(num_samples):
                idx = sample_idx * self.config["sample_frequency"]

                # all tasks should contain the same combination of occ_dems and gt_dems
                # as we are trying to compare different methods
                task_uid = qual_comp_config[0]["task"]
                data_hdf5_group = self.results_hdf5_file[f"task_{task_uid}/{purpose}/data"]
                if ChannelEnum.OCC_DEM.value in data_hdf5_group:
                    occ_dem = data_hdf5_group[ChannelEnum.OCC_DEM.value][idx, ...]
                else:
                    occ_dem = None
                if ChannelEnum.GT_DEM.value in data_hdf5_group:
                    gt_dem = data_hdf5_group[ChannelEnum.GT_DEM.value][idx, ...]
                else:
                    gt_dem = None

                res_grid = data_hdf5_group[ChannelEnum.RES_GRID.value][idx, ...]
                rel_position = data_hdf5_group[ChannelEnum.REL_POSITION.value][idx, ...]

                u = int(round(occ_dem.shape[0] / 2 + rel_position[0] / res_grid[0]))
                v = int(round(occ_dem.shape[1] / 2 + rel_position[1] / res_grid[1]))
                # we only visualize the robot position if its inside the elevation map
                plot_robot_position = 0 < u < occ_dem.shape[0] and 0 < v < occ_dem.shape[1]
                if plot_robot_position:
                    robot_position_pixel = np.array([u, v])
                else:
                    robot_position_pixel = None

                rec_dems = {}
                for task_spec in qual_comp_config:
                    method_title = task_spec["title"]
                    task_uid = task_spec["task"]
                    data_hdf5_group = self.results_hdf5_file[f"task_{task_uid}/{purpose}/data"]

                    comp_dem = data_hdf5_group[ChannelEnum.COMP_DEM.value][idx, ...]
                    rec_dems[method_title] = comp_dem

                draw_qualitative_comparison_plot(idx, purpose_logdir,
                                                 gt_dem=gt_dem, occ_dem=occ_dem, rec_dems=rec_dems,
                                                 robot_position_pixel=robot_position_pixel, remote=self.remote,
                                                 hide_ticks=self.config.get("hide_ticks", False))

                progress_bar.next()
            progress_bar.finish()
