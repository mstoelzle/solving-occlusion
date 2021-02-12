import matplotlib.pyplot as plt
import numpy as np
import pathlib

from src.enums import *


def draw_dataset_samples(sample_idx: int, logdir: pathlib.Path,
                         gt_dem: np.array = None, occ_dem: np.array = None, occ_mask: np.array = None,
                         gt_data_um: np.array = None, occ_data_um: np.array = None,
                         robot_position_pixel: np.array = None, remote=False):
    if occ_dem is None and occ_mask is not None:
        occ_dem = gt_dem.copy()
        occ_dem[occ_mask == 1] = np.nan
    nocc_dem = occ_dem[~np.isnan(occ_dem)]

    dem_vmin = np.min(nocc_dem)
    dem_vmax = np.max(nocc_dem)
    if gt_dem is not None:
        dem_vmin = np.min([dem_vmin, np.min(gt_dem[~np.isnan(gt_dem)])])
        dem_vmax = np.max([dem_vmax, np.max(gt_dem[~np.isnan(gt_dem)])])

    dem_cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[2 * 6.4, 2 * 4.8])

    axes[0, 0].set_title("Occluded DEM")
    # matshow plots x and y swapped
    mat = axes[0, 0].matshow(np.swapaxes(occ_dem, 0, 1), vmin=dem_vmin,
                             vmax=dem_vmax, cmap=dem_cmap)
    fig.colorbar(mat, ax=axes[0, :].ravel().tolist(), fraction=0.045)

    if gt_dem is not None:
        axes[0, 1].set_title("Ground-truth DEM")
        # matshow plots x and y swapped
        mat = axes[0, 1].matshow(np.swapaxes(gt_dem, 0, 1), vmin=dem_vmin,
                                 vmax=dem_vmax, cmap=dem_cmap)

    um_cmap = plt.get_cmap("RdYlGn_r")

    if occ_data_um is not None:
        axes[1, 0].set_title("Occluded data uncertainty")
        # matshow plots x and y swapped
        mat = axes[1, 0].matshow(np.swapaxes(occ_data_um, 0, 1), vmin=0, vmax=4, cmap=um_cmap)
        axes[1, 0].grid(False)

    if gt_data_um is not None:
        axes[1, 1].set_title("Ground-truth data uncertainty")
        # matshow plots x and y swapped
        mat = axes[1, 1].matshow(np.swapaxes(occ_data_um, 0, 1), vmin=0, vmax=4, cmap=um_cmap)
        fig.colorbar(mat, ax=axes[1, :].ravel().tolist(), fraction=0.045)
        axes[1, 1].grid(False)

    for i, ax in enumerate(axes.reshape(-1)):
        if robot_position_pixel is not None:
            ax.plot([robot_position_pixel[0]], [robot_position_pixel[1]], marker="*", color="red")

        # Hide grid lines
        ax.grid(False)

    plt.draw()
    plt.savefig(str(logdir / f"sample_2d_{sample_idx}.pdf"))
    if remote is not True:
        plt.show()
    plt.close()


def draw_error_uncertainty_plot(sample_idx: int, logdir: pathlib.Path,
                                gt_dem: np.array = None, rec_dem: np.array = None, comp_dem: np.array = None,
                                rec_data_um: np.array = None, comp_data_um: np.array = None,
                                model_um: np.array = None, total_um: np.array = None,
                                robot_position_pixel: np.array = None, remote=False, indiv_vranges=False):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[1.3 * 10, 1.9 * 10])

    cmap = plt.get_cmap("RdYlGn_r")

    if indiv_vranges is False:
        error_uncertainty_vmin = np.Inf
        error_uncertainty_vmax = -np.Inf
        if rec_data_um is not None:
            comp_data_um_min = np.min(rec_data_um[~np.isnan(rec_data_um)])
            comp_data_um_max = np.max(rec_data_um[~np.isnan(rec_data_um)])
            error_uncertainty_vmin = np.min([error_uncertainty_vmin, comp_data_um_min])
            error_uncertainty_vmax = np.max([error_uncertainty_vmax, comp_data_um_max])
        if comp_data_um is not None:
            comp_data_um_min = np.min(comp_data_um[~np.isnan(comp_data_um)])
            comp_data_um_max = np.max(comp_data_um[~np.isnan(comp_data_um)])
            error_uncertainty_vmin = np.min([error_uncertainty_vmin, comp_data_um_min])
            error_uncertainty_vmax = np.max([error_uncertainty_vmax, comp_data_um_max])
        if model_um is not None:
            model_um_min = np.min(model_um[~np.isnan(model_um)])
            model_um_max = np.max(model_um[~np.isnan(model_um)])
            error_uncertainty_vmin = np.min([error_uncertainty_vmin, model_um_min])
            error_uncertainty_vmax = np.max([error_uncertainty_vmax, model_um_max])
        if total_um is not None:
            total_um_min = np.min(total_um[~np.isnan(total_um)])
            total_um_max = np.max(total_um[~np.isnan(total_um)])
            error_uncertainty_vmin = np.min([error_uncertainty_vmin, total_um_min])
            error_uncertainty_vmax = np.max([error_uncertainty_vmax, total_um_max])
    else:
        error_uncertainty_vmin = None
        error_uncertainty_vmax = None

    if gt_dem is not None:
        rec_abs_error = np.abs(rec_dem - gt_dem)
        comp_abs_error = np.abs(comp_dem - gt_dem)

        if indiv_vranges is False:
            error_uncertainty_vmin = np.min([error_uncertainty_vmin,
                                             np.min(rec_abs_error[~np.isnan(rec_abs_error)]),
                                             np.min(comp_abs_error[~np.isnan(comp_abs_error)])])
            error_uncertainty_vmax = np.max([error_uncertainty_vmax,
                                             np.max(rec_abs_error[~np.isnan(rec_abs_error)]),
                                             np.max(comp_abs_error[~np.isnan(comp_abs_error)])])

        axes[0, 0].set_title("Reconstruction error")
        # matshow plots x and y swapped
        mat = axes[0, 0].matshow(np.swapaxes(rec_abs_error, 0, 1), cmap=cmap,
                                 vmin=error_uncertainty_vmin, vmax=error_uncertainty_vmax)
        if indiv_vranges:
            fig.colorbar(mat, ax=axes[0, 0], fraction=0.10)
        if robot_position_pixel is not None:
            axes[0, 0].plot(robot_position_pixel[0], robot_position_pixel[1], marker="*", color="blue")
        axes[0, 0].grid(False)

        axes[0, 1].set_title("Composition error")
        # matshow plots x and y swapped
        mat = axes[0, 1].matshow(np.swapaxes(comp_abs_error, 0, 1), cmap=cmap,
                                 vmin=error_uncertainty_vmin, vmax=error_uncertainty_vmax)
        if indiv_vranges:
            fig.colorbar(mat, ax=axes[0, 1], fraction=0.10)
        if robot_position_pixel is not None:
            axes[0, 1].plot(robot_position_pixel[0], robot_position_pixel[1], marker="*", color="blue")
        axes[0, 1].grid(False)

        # axes.append(fig.add_subplot(122, projection="3d"))
        # axes[1].set_title("Reconstruction error")
        # # the np.NaNs in the ground-truth elevation maps give us these warnings:
        # warnings.filterwarnings("ignore", category=UserWarning)
        # axes[1].plot_surface(x_3d, y_3d, np.abs(rec_dem - gt_dem),
        #                      cmap=plt.get_cmap("RdYlGn_r"))
        # warnings.filterwarnings("default", category=UserWarning)
        # axes[1].set_xlabel("x [m]")
        # axes[1].set_ylabel("y [m]")
        # axes[1].set_zlabel("z [m]")

    if rec_data_um is not None:
        axes[1, 0].set_title("Reconstructed data uncertainty")
        # matshow plots x and y swapped
        mat = axes[1, 0].matshow(np.swapaxes(rec_data_um, 0, 1), cmap=cmap,
                                 vmin=error_uncertainty_vmin, vmax=error_uncertainty_vmax)
        if indiv_vranges:
            fig.colorbar(mat, ax=axes[1, 0], fraction=0.10)
        if robot_position_pixel is not None:
            axes[1, 0].plot(robot_position_pixel[0], robot_position_pixel[1], marker="*", color="blue")
        axes[1, 0].grid(False)

    if comp_data_um is not None:
        axes[1, 1].set_title("Composed data uncertainty")
        # matshow plots x and y swapped
        mat = axes[1, 1].matshow(np.swapaxes(comp_data_um, 0, 1), cmap=cmap,
                                 vmin=error_uncertainty_vmin, vmax=error_uncertainty_vmax)
        if indiv_vranges:
            fig.colorbar(mat, ax=axes[1, 1], fraction=0.10)
        if robot_position_pixel is not None:
            axes[1, 1].plot(robot_position_pixel[0], robot_position_pixel[1], marker="*", color="blue")
        axes[1, 1].grid(False)

    if model_um is not None:
        axes[2, 0].set_title("Model uncertainty")
        # matshow plots x and y swapped
        mat = axes[2, 0].matshow(np.swapaxes(model_um, 0, 1), cmap=cmap,
                                 vmin=error_uncertainty_vmin, vmax=error_uncertainty_vmax)
        if indiv_vranges:
            fig.colorbar(mat, ax=axes[2, 0], fraction=0.10)
        if robot_position_pixel is not None:
            axes[2, 0].plot(robot_position_pixel[0], robot_position_pixel[1], marker="*", color="blue")
        axes[2, 0].grid(False)

    if total_um is not None:
        axes[2, 1].set_title("Total uncertainty")
        # matshow plots x and y swapped
        mat = axes[2, 1].matshow(np.swapaxes(total_um, 0, 1), cmap=cmap,
                                 vmin=error_uncertainty_vmin, vmax=error_uncertainty_vmax)
        if indiv_vranges:
            fig.colorbar(mat, ax=axes[2, 1], fraction=0.10)
        if robot_position_pixel is not None:
            axes[2, 1].plot(robot_position_pixel[0], robot_position_pixel[1], marker="*", color="blue")
        axes[2, 1].grid(False)

    if indiv_vranges is False:
        fig.colorbar(mat, ax=axes.ravel().tolist(), fraction=0.1)

    plt.draw()
    plt.savefig(str(logdir / f"error_uncertainty_{sample_idx}.pdf"))
    if remote is not True:
        plt.show()
    plt.close()


def draw_solutions_plot(sample_idx: int, logdir: pathlib.Path,
                        channel: ChannelEnum, dems: np.array,
                        robot_position_pixel: np.array = None, remote=False):
    num_solutions = dems.shape[0]
    grid_size = int(np.floor(np.sqrt(num_solutions)).item())

    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=[2 * 6.4, 2 * 6.4])

    if channel == ChannelEnum.REC_DEMS:
        plt.title("Reconstruction solutions")
    elif channel == ChannelEnum.COMP_DEMS:
        plt.title("Composition solutions")
    else:
        raise NotImplementedError

    dems_vmin = np.min(dems)
    dems_vmax = np.max(dems)

    dems_cmap = plt.get_cmap("viridis")

    solution_idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            mat = axes[i, j].matshow(np.swapaxes(dems[solution_idx, ...], 0, 1),
                                     cmap=dems_cmap, vmin=dems_vmin, vmax=dems_vmax)
            axes[i, j].grid(False)

            solution_idx += 1

    fig.colorbar(mat, ax=axes.ravel().tolist(), fraction=0.045)

    plt.draw()
    plt.savefig(str(logdir / f"{channel.value}_{sample_idx}.pdf"))
    if remote is not True:
        plt.show()
    plt.close()


def draw_traversability_plot(sample_idx: int, logdir: pathlib.Path,
                             gt_dem: np.array = None, rec_dem: np.array = None, comp_dem: np.array = None,
                             rec_data_um: np.array = None, comp_data_um: np.array = None,
                             model_um: np.array = None, total_um: np.array = None,
                             rec_trav_risk_map: np.array = None, comp_trav_risk_map: np.array = None,
                             robot_position_pixel: np.array = None, remote=False):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[1.2 * 10, 1.1 * 10])

    cmap = plt.get_cmap("RdYlGn_r")

    if rec_dem is not None:
        axes[0, 0].set_title("Reconstruction")
        # matshow plots x and y swapped
        mat = axes[0, 0].matshow(np.swapaxes(rec_dem, 0, 1), cmap=cmap)
        fig.colorbar(mat, ax=axes[0, 0], fraction=0.08)

    if comp_dem is not None:
        axes[0, 1].set_title("Composition")
        # matshow plots x and y swapped
        mat = axes[0, 1].matshow(np.swapaxes(comp_dem, 0, 1), cmap=cmap)
        fig.colorbar(mat, ax=axes[0, 1], fraction=0.08)

    if rec_trav_risk_map is not None:
        axes[1, 0].set_title("Reconstructed Traversability")
        # matshow plots x and y swapped
        mat = axes[1, 0].matshow(np.swapaxes(rec_trav_risk_map, 0, 1), cmap=cmap)
        fig.colorbar(mat, ax=axes[1, 0], fraction=0.08)

    if comp_trav_risk_map is not None:
        axes[1, 1].set_title("Composed Traversability")
        # matshow plots x and y swapped
        mat = axes[1, 1].matshow(np.swapaxes(comp_trav_risk_map, 0, 1), cmap=cmap)
        fig.colorbar(mat, ax=axes[1, 1], fraction=0.08)

    for i, ax in enumerate(axes.reshape(-1)):
        if robot_position_pixel is not None:
            ax.plot([robot_position_pixel[0]], [robot_position_pixel[1]], marker="*", color="red")

        # Hide grid lines
        ax.grid(False)

    plt.draw()
    plt.savefig(str(logdir / f"traversability_{sample_idx}.pdf"))
    if remote is not True:
        plt.show()
    plt.close()
