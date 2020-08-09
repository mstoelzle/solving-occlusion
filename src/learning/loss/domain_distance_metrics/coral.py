import torch

from src.utils import get_logger
from src.utils.torch_utils import torch_cov

logger = get_logger("coral")


def coral(data_domain_1: torch.Tensor, data_domain_2: torch.Tensor):
    # Deep CORAL
    # Sun, Baochen, and Kate Saenko. "Deep coral: Correlation alignment for deep domain adaptation."
    # European conference on computer vision. Springer, Cham, 2016.
    # https://github.com/DenisDsh/PyTorch-Deep-CORAL

    assert data_domain_1.size(1) == data_domain_2.size(1)
    d = data_domain_1.size(1)

    # source and target covariance
    sigma_1 = torch_cov(data_domain_1, rowvar=False)
    sigma_2 = torch_cov(data_domain_2, rowvar=False)

    # frobenius norm between source and target
    loss = torch.sum(torch.mul((sigma_1 - sigma_2), (sigma_1 - sigma_2)))
    loss = loss / (4 * d * d)

    return loss
