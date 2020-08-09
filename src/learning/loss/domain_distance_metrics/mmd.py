import torch

from src.utils import get_logger

logger = get_logger("mmd")


# maximum mean discrepancy
# https://github.com/Saswatm123/MMD-VAE/blob/master/MMD_VAE.ipynb
def maximum_mean_discrepancy(a: torch.Tensor, b: torch.Tensor):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2 * gaussian_kernel(a, b).mean()


def gaussian_kernel(a: torch.Tensor, b: torch.Tensor):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2) / depth
    return torch.exp(-numerator)


def gaussian_kernel_2(a: torch.Tensor, b: torch.Tensor, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    num_samples = int(a.size(0)) + int(b.size(0))
    total = torch.cat([a, b], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (num_samples ** 2 - num_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # / len(kernel_val)
