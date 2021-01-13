import getpass
import json
import pathlib
import random
import time
from typing import List

import torch
import numpy as np

from src.utils.log import create_base_logger, create_logdir
from src.utils import measure_runtime, get_git_version
from .experiment import Experiment


class ExperimentSet:
    def __init__(self, **kwargs):
        self.config = kwargs

        self.name = self.config["name"]
        self.experiment_config = self.config["experiment"]
        self.seeds: List[int] = self.config["seeds"]
        self.remote: bool = self.config.get("remote", False)

        self.logdir = create_logdir(f"learning_{self.name}")
        self.create_set_info()

        self.datadir = pathlib.Path("data")
        self.datadir.mkdir(parents=True, exist_ok=True)

        self.logger = create_base_logger(self.logdir, name="experiment_set")

        self.device = torch.device("cuda" if (self.config.get("cuda", True) and torch.cuda.is_available()) else "cpu")

        self.logger.info(f"Using device {self.device}")

    def run(self):
        with measure_runtime(self.logdir):
            for seed in self.seeds:
                torch.cuda.empty_cache()
                self.logger.info(f"Seed {seed} used to run experiment")
                self.set_random_seeds(seed)
                exp_logdir: pathlib.Path = self.create_experiment_logdir(seed)
                experiment: Experiment = Experiment(seed=seed, logdir=exp_logdir, datadir=self.datadir,
                                                    set_name=self.name, device=self.device,
                                                    remote=self.remote, **self.experiment_config)
                experiment.run()
                experiment.plot()

    @staticmethod
    def set_random_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def create_experiment_logdir(self, seed: int) -> pathlib.Path:
        """
        creates a logdir for an Experiment. These logdirs are contained within the ExperimentSet logdir.
        this function also fills the created dir with some basic information about the experiment such as the
        config used and the random seed that was set.
        :param seed:
        :return: the created logdir that should be passed to the Experiment
        """

        exp_logdir: pathlib.Path = self.logdir / f"seed_{seed}"

        # Check that this seed has not been used before
        assert not exp_logdir.exists()

        exp_logdir.mkdir(exist_ok=False, parents=False)

        with open(str(exp_logdir / "config.json"), "w") as fp:
            json.dump(self.experiment_config, fp, indent=4)
        with open(str(exp_logdir / "seed.json"), "w") as fp:
            json.dump({"seed": seed}, fp, indent=4)

        with open(str(exp_logdir / "git_version.json"), "w") as fp:
            json.dump({"version": get_git_version()}, fp, indent=4)

        return exp_logdir

    def create_set_info(self):
        with open(str(self.logdir / "config.json"), "w") as fp:
            json.dump(self.config, fp, indent=4)

        with open(self.logdir / "set.json", "w")as fp:
            json.dump({"name": self.name,
                       "user": getpass.getuser(),
                       "experiment_config": self.experiment_config,
                       "seeds": self.seeds,
                       "start_time": time.time()
                       }, fp)