import numpy as np
import torch
import torch.nn.functional as F

from .mmd import gaussian_kernel_2
from src.utils import get_logger

logger = get_logger("lmmd")


def local_maximum_mean_discrepancy(num_classes,
                                   data_domain_1, data_domain_2, targets_domain_1, pred_targets_domain_2,
                                   kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # Local maximum mean discrepancy (cmmd)
    # Zhu, Yongchun, et al. "Deep Subdomain Adaptation Network for Image Classification."
    # IEEE Transactions on Neural Networks and Learning Systems (2020).
    # https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/mmd.py

    assert data_domain_1.size() == data_domain_2.size()
    assert targets_domain_1.size(0) == pred_targets_domain_2.size(0)

    sample_tensor = data_domain_1
    batch_size = int(sample_tensor.size(0))

    # as the output of the classifier is not guaranteed to be softmaxed, we need to apply a softmax to the predictions
    pred_targets_domain_2 = F.softmax(pred_targets_domain_2, dim=1)

    weight_ss, weight_tt, weight_st = calc_weight(targets_domain_1, pred_targets_domain_2, num_classes)
    weight_ss = torch.from_numpy(weight_ss).to(device=sample_tensor.device)
    weight_tt = torch.from_numpy(weight_tt).to(device=sample_tensor.device)
    weight_st = torch.from_numpy(weight_st).to(device=sample_tensor.device)

    kernels = gaussian_kernel_2(data_domain_1, data_domain_2,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = sample_tensor.new_tensor([0])
    if torch.sum(torch.isnan(sum(kernels))):
        return loss

    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)

    return loss


def convert_to_onehot(sca_label, num_classes):
    return np.eye(num_classes)[sca_label]


def calc_weight(source_labels, target_preds, num_classes):
    batch_size = source_labels.size(0)

    source_labels_cpu = source_labels.detach().cpu()
    target_preds_cpu = target_preds.detach().cpu()

    s_sca_label = source_labels_cpu.data.numpy()
    s_vec_label = convert_to_onehot(s_sca_label, num_classes)
    s_sum = np.sum(s_vec_label, axis=0).reshape(1, num_classes)
    s_sum[s_sum == 0] = 100
    s_vec_label = s_vec_label / s_sum

    t_sca_label = target_preds_cpu.data.max(1)[1].numpy()
    # t_vec_label = convert_to_onehot(t_sca_label)

    t_vec_label = target_preds_cpu.data.numpy()
    t_sum = np.sum(t_vec_label, axis=0).reshape(1, num_classes)
    t_sum[t_sum == 0] = 100
    t_vec_label = t_vec_label / t_sum

    weight_ss = np.zeros((batch_size, batch_size))
    weight_tt = np.zeros((batch_size, batch_size))
    weight_st = np.zeros((batch_size, batch_size))

    set_s = set(s_sca_label)
    set_t = set(t_sca_label)
    count = 0
    for i in range(num_classes):
        if i in set_s and i in set_t:
            s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
            t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
            ss = np.dot(s_tvec, s_tvec.T)
            weight_ss = weight_ss + ss  # / np.sum(s_tvec) / np.sum(s_tvec)
            tt = np.dot(t_tvec, t_tvec.T)
            weight_tt = weight_tt + tt  # / np.sum(t_tvec) / np.sum(t_tvec)
            st = np.dot(s_tvec, t_tvec.T)
            weight_st = weight_st + st  # / np.sum(s_tvec) / np.sum(t_tvec)
            count += 1

    length = count  # len( set_s ) * len( set_t )
    if length != 0:
        weight_ss = weight_ss / length
        weight_tt = weight_tt / length
        weight_st = weight_st / length
    else:
        weight_ss = np.array([0])
        weight_tt = np.array([0])
        weight_st = np.array([0])

    return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
