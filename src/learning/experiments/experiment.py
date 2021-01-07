import pathlib
import json
import traceback

import torch

from src.enums.task_type_enum import TaskTypeEnum
from src.learning.learning_classes.inference import Inference
from src.learning.learning_classes.supervised_learning import SupervisedLearning
from src.learning.tasks import TaskPath
from src.visualization import ResultsPlotter
from src.utils import hash_dict, measure_runtime
from src.utils.log import get_logger
from src.utils.sheet_uploader import SheetUploader


class Experiment:
    def __init__(self, seed: int, logdir: pathlib.Path, datadir: pathlib.Path, set_name: str, device: torch.device,
                 remote: bool = False, **kwargs):
        self.config = kwargs

        self.seed = seed
        self.logdir: pathlib.Path = logdir
        self.datadir: pathlib.Path = datadir
        self.results_hdf5_path: pathlib.Path = self.logdir / "learning_results.hdf5"

        self.logger = get_logger("experiment")
        self.set_name = set_name
        self.hash = hash_dict(kwargs)
        self.save_hash()

        self.device = device

        self.task_path: TaskPath = TaskPath(self.seed, self.logdir, self.datadir, **kwargs["task_path"])

        self.supervised_learning = SupervisedLearning(seed=self.seed, logdir=self.logdir, device=self.device,
                                                      results_hdf5_path=self.results_hdf5_path)

        self.inference = Inference(seed=self.seed, logdir=self.logdir, device=self.device,
                                   results_hdf5_path=self.results_hdf5_path)

        self.results_plotter = ResultsPlotter(results_hdf5_path=self.results_hdf5_path,
                                              logdir=self.logdir, remote=remote,
                                              **self.config.get("visualization", {}))

    def run(self):
        with measure_runtime(self.logdir):
            for task in self.task_path:
                if self.config.get("logging", {}).get("upload", False) is True:
                    self.upload_task()

                with measure_runtime(task.logdir):
                    if task.type == TaskTypeEnum.SUPERVISED_LEARNING:
                        with self.supervised_learning:
                            task.output_model = self.supervised_learning.train(task)
                        self.supervised_learning.reset()
                    elif task.type == TaskTypeEnum.INFERENCE:
                        with self.inference:
                            task.output_model = self.inference.run(task)
                        self.inference.reset()
                    else:
                        raise NotImplementedError(f"The following task type is not implemented: {task.type}")

        if self.config.get("logging", {}).get("upload", False) is True:
            self.upload_task()

        self.save_exit_code(0)

    def upload_task(self):
        try:
            uploader = SheetUploader(self.set_name)
            uploader.upload()

        except Exception:
            exc = traceback.format_exc()
            self.logger.exception(exc)
            self.logger.critical("Could not upload latest results. Likely connection problem to Google sheets.")

    def plot(self):
        self.results_plotter.plot()

    def save_exit_code(self, code: int):
        with open(str(self.logdir / "exit_code.json"), "w") as fp:
            json.dump({"code": code}, fp, indent=4)

    def save_hash(self):
        with open(str(self.logdir / "hash.json"), "w") as fp:
            json.dump({"hash": self.hash}, fp, indent=4)
