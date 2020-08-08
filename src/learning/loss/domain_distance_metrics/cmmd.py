import torch

from .mmd import gaussian_kernel_2
from path_learning.utils import get_logger

logger = get_logger("cmmd")


def conditional_maximum_mean_discrepancy(num_classes,
                                         data_domain_1, data_domain_2, targets_domain_1, targets_domain_2,
                                         kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # Conditional maximum mean discrepancy (cmmd)
    # Zhu, Yongchun, et al. "Multi-representation adaptation network for cross-domain image classification."
    # Neural Networks 119 (2019): 214-221.
    # https://github.com/jindongwang/transferlearning/blob/master/code/deep/MRAN/mmd.py

    assert data_domain_1.size() == data_domain_2.size()
    assert targets_domain_1.size() == targets_domain_2.size()

    sample_tensor = data_domain_1
    batch_size = int(sample_tensor.size(0))

    s_label = targets_domain_1.view(batch_size, 1)
    s_label = sample_tensor.new_zeros(batch_size, num_classes).scatter_(1, s_label.data, 1)

    t_label = targets_domain_2.view(batch_size, 1)
    t_label = sample_tensor.new_zeros(batch_size, num_classes).scatter_(1, t_label.data, 1)

    kernels = gaussian_kernel_2(data_domain_1, data_domain_2,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    loss += torch.mean(torch.mm(s_label, torch.transpose(s_label, 0, 1)) * XX +
                       torch.mm(t_label, torch.transpose(t_label, 0, 1)) * YY -
                       2 * torch.mm(s_label, torch.transpose(t_label, 0, 1)) * XY)

    return loss
