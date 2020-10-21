from copy import deepcopy
import pathlib
from typing import Dict, List, Optional

import torch

from src.enums.task_type_enum import TaskTypeEnum
from src.dataloaders.dataloader import Dataloader
from .task import Task
from src.utils.log import get_logger

logger = get_logger("task_path")


class TaskPath:
    """
    Task paths are iterables that return a task and increasing uid at each iteration. The TaskPath also measures
    the runtime of each of its path by logging the time between calls itself (the task iterator).
    """
    def __init__(self, logdir: pathlib.Path, datadir: pathlib.Path, **kwargs):
        self.config = kwargs

        self.task_configs = kwargs["tasks"]
        self.default_values = kwargs["defaults"]

        self.idx: int = 0
        self.logdir: pathlib.Path = logdir
        self.datadir: pathlib.Path = datadir

        self.tasks: List[Task] = []

    def get_next_config(self) -> Dict:
        task_config: Dict = self.task_configs[self.idx]

        # update task level with defaults from task_path
        recursive_default_setting(self.default_values, task_config)

        # update domain with defaults from task level domain defaults
        for domain, domain_config in task_config['domains'].items():
            recursive_default_setting(task_config["domain_defaults"], domain_config)

        logger.info(f"Loaded defaults for task {self.idx}")
        return task_config

    def get_next_logdir(self) -> pathlib.Path:
        task_logdir = self.logdir / f"task_{self.idx}"
        task_logdir.mkdir(parents=False, exist_ok=True)
        return task_logdir

    def init_next_task(self):
        task_logdir = self.get_next_logdir()
        task_config = self.get_next_config()

        task = Task(self.idx, task_logdir, **task_config)

        for model_to_pick in ["model_to_train", "model_to_infer"]:
            if model_to_pick in task.config:
                if task.config[model_to_pick] == 'untrained':
                    # we need to initialize the model from the learning map config in the base learner
                    setattr(task, model_to_pick, None)
                elif task.config[model_to_pick] == 'pretrained':
                    setattr(task, model_to_pick, 'pretrained')
                elif type(task.config[model_to_pick]) == str and pathlib.Path(task.config[model_to_pick]).exists():
                    setattr(task, model_to_pick,  pathlib.Path(task.config[model_to_pick]))
                else:
                    if type(task.config[model_to_pick]) == int:
                        model = deepcopy(self.tasks[task.config[model_to_pick]].output_model)
                    elif task.config[model_to_pick] == 'prior_task':
                        model = deepcopy(self.get_prior_model())
                    elif task.config[model_to_pick] == 'prior_supervised_learning':
                        model = deepcopy(self.get_prior_model(TaskTypeEnum.SUPERVISED_LEARNING))
                    else:
                        raise ValueError(f"Cannot interpret the {model_to_pick} config parameter: "
                                         f"{task.config['model_to_pick']} for task {self.idx}")

                    if model is None:
                        ValueError(f"The model could not get properly assigned for {model_to_pick}")
                    setattr(task, model_to_pick, model)

        for dataloader_type, dataloader_spec in task.config['dataloaders'].items():
            if task.type == TaskTypeEnum.INFERENCE:
                dataloader = Dataloader(purposes=["test"], **task.config["domains"][dataloader_spec])
            else:
                dataloader = Dataloader(**task.config["domains"][dataloader_spec])

            if dataloader is None:
                ValueError(f"The {dataloader_type} could not get properly assigned")
            setattr(task, dataloader_type, dataloader)

        self.tasks.append(task)

    def get_prior_model(self, task_type: TaskTypeEnum = None) -> Optional[torch.nn.Module]:
        for task in reversed(self.tasks):
            if task_type is None:
                return task.output_model
            else:
                if task.type == task_type:
                    return task.output_model

        return None

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> Task:

        if len(self.tasks) > 0:
            # save a positive exit code for the task that just finished.
            self.tasks[-1].save_exit_code(0)

            self.tasks[-1].cleanup()

        if self.idx >= len(self.task_configs):
            raise StopIteration

        self.init_next_task()
        task = self.tasks[-1]

        self.idx += 1

        return task


# the source config is the defaults and the target config is the task config
def recursive_default_setting(source_config: dict, target_config: dict):
    for default_key, default_value in source_config.items():
        if default_key not in target_config:
            target_config.update({default_key: default_value})
        elif isinstance(default_value, dict):
            recursive_default_setting(source_config[default_key], target_config[default_key])
