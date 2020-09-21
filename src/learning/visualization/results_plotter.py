import h5py
import numpy
import pathlib
import seaborn as sb
import torch
from typing import *


class ResultsPlotter:
    def __init__(self, results_hdf5_path: pathlib.Path, **kwargs):
        self.config = kwargs

        self.results_hdf5_path = results_hdf5_path

        self.results_hdf5_file: Optional[h5py.File] = None

    def plot(self):
        self.results_hdf5_file: h5py.File = h5py.File(str(self.results_hdf5_path), 'r')
        self.results_hdf5_file.__enter__()

        for task_string, task_group in self.results_hdf5_file.items():
            task_string_split = task_string.split("_")
            task_uid = task_string_split[1]


        self.results_hdf5_file.__exit__()
