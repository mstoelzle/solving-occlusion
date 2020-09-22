import h5py
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import seaborn as sb
import torch
from typing import *

from src.enums import *
from src.utils.log import get_logger

logger = get_logger("results_plotter")


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

                self.plot_task(task_uid, "test", task_hdf5_group)

    def plot_task(self, task_uid: int, purpose: str, task_hdf5_group: h5py.Group, ):
        logger.info(f"Plot task {task_uid}")
        if self.config.get("sample_frequency", 0) > 0:
            samples_dir = self.logdir / f"task_{task_uid}" / f"{purpose}_samples"
            samples_dir.mkdir(exist_ok=True, parents=True)
            data_hdf5_group = task_hdf5_group[f"{purpose}/data"]
            num_samples = int(len(data_hdf5_group[ChannelEnum.ELEVATION_MAP.value])/self.config["sample_frequency"])
            for sample_idx in range(num_samples):
                idx = sample_idx * self.config["sample_frequency"]
                elevation_map = data_hdf5_group[ChannelEnum.ELEVATION_MAP.value][idx, ...]
                reconstructed_elevation_map = data_hdf5_group[ChannelEnum.RECONSTRUCTED_ELEVATION_MAP.value][idx, ...]
                occluded_elevation_map = data_hdf5_group[ChannelEnum.OCCLUDED_ELEVATION_MAP.value][idx, ...]

                fig, axes = plt.subplots(nrows=1, ncols=3)
                axes[0].set_title("Ground-truth")
                mat = axes[0].matshow(np.swapaxes(elevation_map, 0, 1))  # matshow plots x and y swapped
                axes[1].set_title("Reconstruction")
                mat = axes[1].matshow(np.swapaxes(reconstructed_elevation_map, 0, 1))  # matshow plots x and y swapped
                axes[2].set_title("Occluded")
                mat = axes[2].matshow(np.swapaxes(occluded_elevation_map, 0, 1))  # matshow plots x and y swapped

                fig.colorbar(mat, ax=axes.ravel().tolist(), fraction=0.021)

                plt.draw()
                plt.savefig(str(samples_dir / f"sample_{idx}.pdf"))
                if self.remote is not True:
                    plt.show()
