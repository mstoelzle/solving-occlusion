import cv2 as cv
import torch

from .base_baseline_model import BaseBaselineModel


class OpenCVBaseline(BaseBaselineModel):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)

        if name == "NavierStokes":
            self.inpainting_method = cv.INPAINT_NS
        elif name == "Telea":
            self.inpainting_method = cv.INPAINT_TELEA
        else:
            raise ValueError

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass