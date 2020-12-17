import matplotlib.pyplot as plt
import numpy as np
import pathlib

from src.enums import *


def draw_error_uncertainty_plot(sample_idx: int, logdir: pathlib.Path,
                                gt_dem: np.array = None, rec_dem: np.array = None, comp_dem: np.array = None,
                                model_uncertainty_map: np.array = None, data_uncertainty_map: np.array = None,
                                robot_position: np.array = None, remote=False, indiv_vranges=False):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[10, 10])

    cmap = plt.get_cmap("RdYlGn_r")

    if indiv_vranges is False:
        error_uncertainty_vmin = np.Inf
        error_uncertainty_vmax = -np.Inf
        if data_uncertainty_map is not None:
            data_uncertainty_min = np.min(data_uncertainty_map[~np.isnan(data_uncertainty_map)])
            data_uncertainty_max = np.max(data_uncertainty_map[~np.isnan(data_uncertainty_map)])
            error_uncertainty_vmin = np.min([error_uncertainty_vmin, data_uncertainty_min])
            error_uncertainty_vmax = np.max([error_uncertainty_vmax, data_uncertainty_max])
        if model_uncertainty_map is not None:
            model_uncertainty_min = np.min(model_uncertainty_map[~np.isnan(model_uncertainty_map)])
            model_uncertainty_max = np.max(model_uncertainty_map[~np.isnan(model_uncertainty_map)])
            error_uncertainty_vmin = np.min([error_uncertainty_vmin, model_uncertainty_min])
            error_uncertainty_vmax = np.max([error_uncertainty_vmax, model_uncertainty_max])
    else:
        error_uncertainty_vmin = None
        error_uncertainty_vmax = None

    if gt_dem is not None:
        rec_abs_error = np.abs(rec_dem - gt_dem)
        comp_abs_error = np.abs(comp_dem - gt_dem)

        if indiv_vranges is False:
            error_uncertainty_vmin = np.min([error_uncertainty_vmin,
                                             np.min(rec_abs_error), np.min(comp_abs_error)])
            error_uncertainty_vmax = np.max([error_uncertainty_vmax,
                                             np.max(rec_abs_error), np.max(comp_abs_error)])

        axes[0, 0].set_title("Reconstruction error")
        # matshow plots x and y swapped
        mat = axes[0, 0].matshow(np.swapaxes(rec_abs_error, 0, 1), cmap=cmap,
                                 vmin=error_uncertainty_vmin, vmax=error_uncertainty_vmax)
        if indiv_vranges:
            fig.colorbar(mat, ax=axes[0, 0], fraction=0.10)
        if robot_position is not None:
            axes[0, 0].plot(robot_position[0], robot_position[1], marker="*", color="blue")
        axes[0, 0].grid(False)

        axes[0, 1].set_title("Composition error")
        # matshow plots x and y swapped
        mat = axes[0, 1].matshow(np.swapaxes(comp_abs_error, 0, 1), cmap=cmap,
                                 vmin=error_uncertainty_vmin, vmax=error_uncertainty_vmax)
        if indiv_vranges:
            fig.colorbar(mat, ax=axes[0, 1], fraction=0.10)
        if robot_position is not None:
            axes[0, 1].plot(robot_position[0], robot_position[1], marker="*", color="blue")
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

    if data_uncertainty_map is not None:
        axes[1, 0].set_title("Data uncertainty")
        # matshow plots x and y swapped
        mat = axes[1, 0].matshow(np.swapaxes(data_uncertainty_map, 0, 1), cmap=cmap,
                                 vmin=error_uncertainty_vmin, vmax=error_uncertainty_vmax)
        if indiv_vranges:
            fig.colorbar(mat, ax=axes[1, 0], fraction=0.10)
        if robot_position is not None:
            axes[1, 0].plot(robot_position[0], robot_position[1], marker="*", color="blue")
        axes[1, 0].grid(False)

    if model_uncertainty_map is not None:
        axes[1, 1].set_title("Model uncertainty")
        # matshow plots x and y swapped
        mat = axes[1, 1].matshow(np.swapaxes(model_uncertainty_map, 0, 1), cmap=cmap,
                                 vmin=error_uncertainty_vmin, vmax=error_uncertainty_vmax)
        if indiv_vranges:
            fig.colorbar(mat, ax=axes[1, 1], fraction=0.10)
        if robot_position is not None:
            axes[1, 1].plot(robot_position[0], robot_position[1], marker="*", color="blue")
        axes[1, 1].grid(False)

    if indiv_vranges is False:
        fig.colorbar(mat, ax=axes.ravel().tolist(), fraction=0.045)

    plt.draw()
    plt.savefig(str(logdir / f"error_uncertainty_{sample_idx}.pdf"))
    if remote is not True:
        plt.show()
    plt.close()


def draw_solutions_plot(sample_idx: int, logdir: pathlib.Path,
                        channel: ChannelEnum, dems: np.array,
                        robot_position: np.array = None, remote=False):

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

