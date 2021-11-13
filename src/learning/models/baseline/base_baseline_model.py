from src.learning.models.base_model import BaseModel


class BaseBaselineModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(strict_forward_def=False, **kwargs)
