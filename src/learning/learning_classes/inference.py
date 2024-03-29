from src.learning.learning_classes.base_learning import BaseLearning
from src.learning.tasks import Task
from src.utils.log import get_logger

logger = get_logger("inference")


class Inference(BaseLearning):
    def __init__(self, **kwargs):
        super().__init__(logger=logger, **kwargs)

    def run(self, task: Task):
        self.task = task

        self.set_model(task.model_to_infer, pick_optimizer=False)
        if self.task.config.get("model", {}).get("trace", True) and self.model.strict_forward_def is True:
            self.trace_model()

        return self.infer()
