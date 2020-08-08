import pathlib
import json
import logging
import subprocess
import time
from typing import Dict

from .log import ROOT_DIR, get_logger

LIB_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
CACHE_DIR = ROOT_DIR / "cache"

logger = get_logger("utils")


def project_path(relative_path: str):
    return LIB_DIR / relative_path


def create_dir(dirname):
    p = dirname
    if not (p.exists() and p.is_dir()):
        p.mkdir(parents=True)
    else:
        print(f"Folder already exists")
    return p


class MeasureRuntime:
    def __init__(self, logdir: pathlib.Path):
        self.logdir: pathlib.Path = logdir

    def __enter__(self):
        self.start_time: float = time.time()
        with open(str(self.logdir / "runtime.json"), "w") as fp:
            json.dump({"start_time": self.start_time}, fp)

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time: float = time.time()
        time_delta: int = int(end_time - self.start_time)

        with open(str(self.logdir / "runtime.json"), "w") as fp:
            json.dump({"start_time": self.start_time, "end_time": end_time, "duration": time_delta}, fp)


def measure_runtime(logdir: pathlib.Path) -> MeasureRuntime:
    return MeasureRuntime(logdir)


def hash_dict(dictionary: Dict) -> int:
    serial = json.dumps(dictionary, sort_keys=True)
    return hash(serial)


def get_git_version() -> str:
    try:
        return subprocess.check_output(["git", "describe"], stderr=subprocess.STDOUT).strip().decode("utf-8")
    except subprocess.CalledProcessError:
        logger.warning("Could not parse git version.")
        return "error"
