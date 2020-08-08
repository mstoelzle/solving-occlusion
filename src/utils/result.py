import pathlib
import json
import math
import datetime
import traceback
from typing import Dict, Optional, List, Union

import pandas as pd
import torch

from .log import get_logger
from src.learning.models.models import pick_model
from src.learning.loss.supervised_learning_loss import SupervisedLearningLoss
from src.learning.tasks import Task


logger = get_logger("results")


class _Result:

    excepted_exceptions = (FileNotFoundError, KeyError, ValueError, AssertionError)

    def __init__(self, result_type: str, result_dir):
        self.dir: pathlib.Path = result_dir
        self.type: str = result_type

        self.complete: bool = True
        self.exit_code: Optional[int] = None
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.start_date_str = None
        self.start_time_str = None

        try:
            self.load_start_time_info()
        except self.excepted_exceptions:
            logger.warning("Could not yet load start time information")
            pass

    def load_exit_code(self) -> None:
        self.exit_code = self.parse_json(self.dir / "exit_code.json").get("code", 1)

    def load_end_time_info(self):
        runtime = self.parse_json(self.dir / "runtime.json")
        self.load_start_time_info()
        self.end_time: float = float(runtime.get("end_time", 0))
        self.duration: float = float(runtime.get("duration", 0))

    def load_start_time_info(self):
        runtime = self.parse_json(self.dir / "runtime.json")
        start_datetime = datetime.datetime.fromtimestamp(int(runtime.get("start_time", 0)))
        self.start_time: float = float(runtime.get("start_time", 0))
        self.start_date_str: str = start_datetime.strftime("%Y/%m/%d")
        self.start_time_str: str = start_datetime.strftime("%H:%M:%S")

    def check_completion(self, exit_code: bool = True, runtime: bool = True):
        try:
            self.load_start_time_info()
            self.load_exit_code()
            self.load_end_time_info()
        except (FileNotFoundError, KeyError):
            logger.info(f"Could not load exit code or runtime from {self.dir}"
                        f" - this might be because object is still running.")
            self.complete = False

    @staticmethod
    def parse_json(path: pathlib.Path):
        with open(str(path), "r") as fp:
            return json.load(fp)


class TaskResult(_Result):
    def __init__(self, task_dir: pathlib.Path):

        try:
            super().__init__("task", task_dir)

            info = self.parse_json(self.dir / "info.json")
            self.config: Dict = info["config"]
            self.uid: int = info["uid"]
            self.dataset = info["config"]["domains"]["target"]["dataset"]

            self.best_epoch: Optional[int] = None
            self.val_loss: Optional[float] = None
            self.val_loss_blame: Optional[List] = None
            self.val_acc: Optional[float] = None
            self.val_acc_per_class: Optional[List] = None
            self.train_loss: Optional[float] = None
            self.train_loss_blame: Optional[List] = None
            self.train_acc: Optional[float] = None
            self.train_acc_per_class: Optional[List] = None
            self.test_loss: Optional[float] = None
            self.test_loss_blame: Optional[List] = None
            self.test_acc: Optional[float] = None
            self.test_acc_per_class: Optional[List] = None

            self.check_completion()
            if self.exit_code == 0:
                self.load_final_results()

        except self.excepted_exceptions:
            logger.warning(traceback.format_exc())
            self.complete = False

    def __str__(self):
        out = f"task {self.uid}\n"
        for key in self.__dict__:
            if key == "config":
                out += f"|\t|\t|\tconfig: ...\n"
            else:
                out += f"|\t|\t|\t{key}: {self.__dict__[key]}\n"
        return out

    def load_final_results(self):
        best_results = self.find_best_results()
        self.best_epoch: int = best_results["best_epoch"]
        self.val_loss: float = best_results["val_loss"]
        self.val_loss_blame: List = best_results["val_loss_blame"]
        self.val_acc: float = best_results["val_acc"]
        self.val_acc_per_class: List = best_results["val_acc_per_class"]
        self.train_loss: float = best_results["train_loss"]
        self.train_loss_blame: List = best_results["train_loss_blame"]
        self.train_acc: float = best_results["train_acc"]
        self.train_acc_per_class: List = best_results["train_acc_per_class"]
        self.test_loss: float = best_results["test_loss"]
        self.test_loss_blame: List = best_results["test_loss_blame"]
        self.test_acc: float = best_results["test_acc"]
        self.test_acc_per_class: List = best_results["test_acc_per_class"]

    def load_all_results(self) -> Dict[str, pd.DataFrame]:
        return {purpose: pd.read_csv(self.dir / f"{purpose}_losses.csv") for purpose in ("train", "val", "test")}

    def find_best_results(self) -> Dict:

        def load_acc(acc: Union[str, float]) -> Union[float, None]:
            return None if math.isnan(float(acc)) else float(acc)

        def load_acc_per_class(series: pd.Series) -> List:
            acc_per_class = series.loc[series.index.str.startswith('class_accuracy_')]
            return acc_per_class.tolist()

        def load_loss_blame(series: pd.Series) -> List:
            loss_blame = series.loc[series.index.str.startswith('loss_blame_')]
            return loss_blame.tolist()

        best_results = {}
        try:
            results = self.load_all_results()

            # get the index of the best validation result
            idx = results["val"]['loss'].idxmin()
            best_val_result = results["val"].iloc[idx]

            best_results["best_epoch"] = int(best_val_result["epoch"])
            best_results["val_loss"] = float(best_val_result["loss"])
            best_results["val_loss_blame"] = load_loss_blame(best_val_result)
            best_results["val_acc"] = load_acc(best_val_result["accuracy"])
            best_results["val_acc_per_class"] = load_acc_per_class(best_val_result)

            if best_results["best_epoch"] != -1:
                train_results = results["train"]

                best_train_result = train_results.loc[train_results["epoch"] == best_val_result["epoch"]]
                if len(best_train_result) > 1:
                    raise ValueError(f"Found more than one result for epoch {train_results['epoch'] } "
                                     f"in {self.dir / 'train_losses.csv'}")

                # squeeze to pandas series
                best_train_result = best_train_result.squeeze(axis=0)

                best_results["train_loss"] = float(best_train_result["loss"])
                best_results["train_loss_blame"] = load_loss_blame(best_train_result)
                best_results["train_acc"] = load_acc(best_train_result["accuracy"])
                best_results["train_acc_per_class"] = load_acc_per_class(best_train_result)
            else:
                best_results["train_loss"] = None
                best_results["train_loss_blame"] = None
                best_results["train_acc"] = None
                best_results["train_acc_per_class"] = None

            test_result = results["test"].loc[results["test"]["epoch"] == 0].squeeze(axis=0)
            best_results["test_loss"] = float(test_result["loss"])
            best_results["test_loss_blame"] = load_loss_blame(test_result)
            best_results["test_acc"] = load_acc(test_result["accuracy"])
            best_results["test_acc_per_class"] = load_acc_per_class(test_result)

        except Exception:
            logger.info(f"Could not gather final results for task in  {self.dir}")

        return best_results

    def generate_task(self, logdir):
        return Task(self.uid, logdir=logdir **self.config)

    def generate_loss(self, logdir):
        logdir = logdir / "tmp_loss"
        logdir.mkdir(exist_ok=True, parents=False)
        return SupervisedLearningLoss(logdir, **self.config["loss"])


class ExperimentResult(_Result):
    def __init__(self, exp_dir: pathlib.Path):

        self.tasks: List[TaskResult] = []
        try:

            super().__init__("experiment", exp_dir)

            self.seed: int = self.parse_json(self.dir / "seed.json")["seed"]
            self.hash: int = self.parse_json(self.dir / "hash.json")["hash"]
            self.git_version: str = self.parse_json(self.dir / "git_version.json")["version"]

            self.config: Dict = self.parse_json(self.dir / "config.json")
            self.tasks = self.load_tasks()

            self.check_completion()

        except self.excepted_exceptions:
            logger.warning(traceback.format_exc())
            self.complete = False

        finally:
            self.complete = self.complete and all([task.complete for task in self.tasks])

    def load_tasks(self) -> List[TaskResult]:
        tasks = [TaskResult(subdir) for subdir in self.dir.iterdir() if "task" in subdir.stem and subdir.is_dir()]
        assert len(tasks) > 0, f"Experiment {self.dir} did not contain any valid tasks."
        return tasks

    def load_model(self, task_uid: Optional[int] = None) -> torch.nn.Module:
        model = pick_model(**self.config["learn_mapping"]["model"])
        if task_uid is not None:
            matched_task = next(filter(lambda task: task.uid == task_uid, self.tasks))
            model.load_state_dict(torch.load(matched_task.dir / "model.pt"))
        return model

    def __str__(self):
        out = f"experiment {self.seed}\n"
        for key in self.__dict__:
            if key == "tasks":
                out += f"|\t|\ttasks: \n"
                for task in self.tasks:
                    out += f"|\t|\t{str(task)}"
            elif key == "config":
                out += f"|\t|\tconfig: ...\n"
            else:
                out += f"|\t|\t{key}: {self.__dict__[key]}\n"

        return out


class ExperimentSetResult(_Result):
    def __init__(self, exp_set_dir: pathlib.Path):
        self.experiments: List[ExperimentResult] = []
        try:
            super().__init__("experiment_set", exp_set_dir)
            set_info = self.parse_json(self.dir / "set.json")
            self.name: str = set_info["name"]
            self.user: str = set_info["user"]
            self.seeds: List[int] = set_info["seeds"]
            self.max_n_processes_per_gpu: int = set_info["max_n_processes_per_gpu"]
            self.experiments = self.get_experiments()

            self.check_completion(exit_code=False, runtime=True)

        except self.excepted_exceptions:
            logger.warning(traceback.format_exc())
            self.complete = False

        finally:
            self.complete = self.complete and all([experiment.complete for experiment in self.experiments])

    def get_experiments(self) -> List[ExperimentResult]:
        experiments = [ExperimentResult(subdir) for subdir in self.dir.iterdir()
                       if "seed" in subdir.stem and subdir.is_dir()]
        assert len(experiments) > 0, f"Experiment {self.dir} did not contain any valid experiments."
        return experiments

    def __str__(self):
        out = f"{self.type}\n"
        for key in self.__dict__:
            if key != "experiments":
                out += f"|\t{key}: {self.__dict__[key]}\n"
            else:
                out += f"|\texperiments: \n"
                for experiment in self.experiments:
                    out += f"|\t{str(experiment)}"
        return out

