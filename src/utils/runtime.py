import json
import pathlib
import statistics
import time
from typing import Optional


class Runtime:
    def __init__(self, name: str, log=False, logdir: Optional[pathlib.Path] = None):
        self.name = name
        self.log = log
        self.logdir: Optional[pathlib.Path] = logdir

    def __enter__(self):
        self.start_time: float = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time: float = time.time()
        self.duration = self.end_time - self.start_time

        if self.log:
            with open(str(self.logdir / f"runtime_{self.name}.json"), "w") as fp:
                json.dump({"start_time": self.start_time, "end_time": self.end_time, "duration": self.duration}, fp)


def log_runtime(name: str, logdir: pathlib.Path) -> Runtime:
    return Runtime(name, log=True, logdir=logdir)


class RuntimeStatistics:
    def __init__(self, name: str, batch_size=1, log=False, logdir: Optional[pathlib.Path] = None):
        self.name = name
        self.log = log
        self.logdir: Optional[pathlib.Path] = logdir
        self.batch_size = batch_size

        self.runtimes = []
        self.durations = []

        self.duration_mean = 0
        self.duration_stdev = 0

    def __enter__(self):
        self.runtimes.append(Runtime(self.name, log=False))
        self.runtimes[-1].__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.runtimes[-1].__exit__(exc_type, exc_val, exc_tb)
        self.durations.append(self.runtimes[-1].duration / self.batch_size)

    def __getitem(self, item):
        return self.runtime[item]

    def collect_statistics(self):
        self.duration_mean = statistics.mean(self.durations)
        self.duration_stdev = statistics.stdev(self.durations)

        if self.log:
            with open(str(self.logdir / f"runtime_statistics_{self.name}.json"), "w") as fp:
                json.dump({"duration_mean": self.duration_mean, "duration_stdev": self.duration_stdev,
                           "samples": len(self.durations)}, fp)

        return self.duration_mean, self.duration_stdev