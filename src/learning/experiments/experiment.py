import pathlib
import json
import traceback

import torch

from src.learning.domain_confusion_learning import DomainConfusionLearning
from src.learning.supervised_learning import SupervisedLearning
from src.learning.tasks import TaskPath
from src.utils import hash_dict, measure_runtime
from src.utils.log import get_logger
from src.utils.sheet_uploader import SheetUploader


class Experiment:
    def __init__(self, logdir: pathlib.Path, datadir: pathlib.Path, set_name: str, device: torch.device, **kwargs):
        self.config = kwargs

        self.logdir: pathlib.Path = logdir
        self.datadir: pathlib.Path = datadir

        self.logger = get_logger("experiment")
        self.set_name = set_name
        self.hash = hash_dict(kwargs)
        self.save_hash()

        self.device = device

        self.task_path: TaskPath = TaskPath(self.logdir, self.datadir, **kwargs["task_path"])

        self.domain_confusion_learning = DomainConfusionLearning(logdir=self.logdir, device=self.device)
        self.supervised_learning = SupervisedLearning(logdir=self.logdir, device=self.device)

    def run(self):
        with measure_runtime(self.logdir):
            for task in self.task_path:
                if self.config.get("logging", {}).get("upload", False) is True:
                    self.upload_task()

                with measure_runtime(task.logdir):
                    if task.type == 'supervised-learning':
                        task.output_model = self.supervised_learning.train(task)
                        self.supervised_learning.reset()
                    elif task.type == 'semi-supervised-learning':
                        task.output_model = self.semi_supervised_learning.train(task)
                        self.semi_supervised_learning.reset()
                    elif task.type == 'domain-confusion':
                        task.output_model = self.domain_confusion_learning.train(task)
                        self.domain_confusion_learning.reset()
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

    def save_exit_code(self, code: int):
        with open(str(self.logdir / "exit_code.json"), "w") as fp:
            json.dump({"code": code}, fp, indent=4)

    def save_hash(self):
        with open(str(self.logdir / "hash.json"), "w") as fp:
            json.dump({"hash": self.hash}, fp, indent=4)
