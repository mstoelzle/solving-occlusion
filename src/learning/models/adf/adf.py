import operator
from collections import OrderedDict
from itertools import islice

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _ConvTransposeMixin
from torch.nn.modules.utils import _pair

from .interpolation import resize2D_as
from .math import normpdf, normcdf


def keep_variance(x, min_variance):
    return x + min_variance


class AvgPool2d(nn.AvgPool2d):
    def __init__(self, keep_variance_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._keep_variance_fn = keep_variance_fn

    def forward(self, inputs_mean, inputs_variance) -> list:
        outputs_mean = F.avg_pool2d(inputs_mean, self.kernel_size, self.stride,
                                    self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
        outputs_variance = F.avg_pool2d(inputs_variance, self.kernel_size, self.stride,
                                        self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
        outputs_variance = outputs_variance / (inputs_mean.size(2) * inputs_mean.size(3))

        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)

        # TODO: avg pooling means that every neuron is multiplied by the same
        #       weight, that is 1/number of neurons in the channel
        #      outputs_variance*1/(H*W) should be enough already

        return [outputs_mean, outputs_variance]


class Softmax(nn.Softmax):
    def __init__(self, keep_variance_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance, eps=1e-5) -> list:
        """Softmax function applied to a multivariate Gaussian distribution.
        It works under the assumption that features_mean and features_variance
        are the parameters of a the indepent gaussians that contribute to the
        multivariate gaussian.
        Mean and variance of the log-normal distribution are computed following
        https://en.wikipedia.org/wiki/Log-normal_distribution."""

        log_gaussian_mean = features_mean + 0.5 * features_variance
        log_gaussian_variance = 2 * log_gaussian_mean

        log_gaussian_mean = torch.exp(log_gaussian_mean)
        log_gaussian_variance = torch.exp(log_gaussian_variance)
        log_gaussian_variance = log_gaussian_variance * (torch.exp(features_variance) - 1)

        constant = torch.sum(log_gaussian_mean, dim=self.dim) + eps
        constant = constant.unsqueeze(self.dim)
        outputs_mean = log_gaussian_mean / constant
        outputs_variance = log_gaussian_variance / (constant ** 2)

        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return [outputs_mean, outputs_variance]


class ReLU(nn.ReLU):
    def __init__(self, keep_variance_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance) -> list:
        features_stddev = torch.sqrt(features_variance)
        div = features_mean / features_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        outputs_mean = features_mean * cdf + features_stddev * pdf
        outputs_variance = (features_mean ** 2 + features_variance) * cdf \
                           + features_mean * features_stddev * pdf - outputs_mean ** 2
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return [outputs_mean, outputs_variance]


class LeakyReLU(nn.LeakyReLU):
    def __init__(self, keep_variance_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance) -> list:
        features_stddev = torch.sqrt(features_variance)
        div = features_mean / features_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        negative_cdf = 1.0 - cdf
        mu_cdf = features_mean * cdf
        stddev_pdf = features_stddev * pdf
        squared_mean_variance = features_mean ** 2 + features_variance
        mean_stddev_pdf = features_mean * stddev_pdf
        mean_r = mu_cdf + stddev_pdf
        variance_r = squared_mean_variance * cdf + mean_stddev_pdf - mean_r ** 2
        mean_n = - features_mean * negative_cdf + stddev_pdf
        variance_n = squared_mean_variance * negative_cdf - mean_stddev_pdf - mean_n ** 2
        covxy = - mean_r * mean_n
        outputs_mean = mean_r - self._negative_slope * mean_n
        outputs_variance = variance_r \
                           + self._negative_slope * self._negative_slope * variance_n \
                           - 2.0 * self._negative_slope * covxy
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return [outputs_mean, outputs_variance]


class Dropout2d(nn.Dropout2d):
    def __init__(self, keep_variance_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._keep_variance_fn = keep_variance_fn

    def forward(self, inputs_mean, inputs_variance) -> list:
        if self.training:
            binary_mask = torch.ones_like(inputs_mean)
            binary_mask = F.dropout2d(binary_mask, self.p, self.training, self.inplace)

            outputs_mean = inputs_mean * binary_mask
            outputs_variance = inputs_variance * binary_mask ** 2

            if self._keep_variance_fn is not None:
                outputs_variance = self._keep_variance_fn(outputs_variance)
            return outputs_mean, outputs_variance

        outputs_variance = inputs_variance
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return [inputs_mean, outputs_variance]


class MaxPool2d(nn.MaxPool2d):
    def __init__(self, keep_variance_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._keep_variance_fn = keep_variance_fn

    def _max_pool_internal(self, mu_a, mu_b, var_a, var_b):
        stddev = torch.sqrt(var_a + var_b)
        ab = mu_a - mu_b
        alpha = ab / stddev
        pdf = normpdf(alpha)
        cdf = normcdf(alpha)
        z_mu = stddev * pdf + ab * cdf + mu_b
        z_var = ((mu_a + mu_b) * stddev * pdf +
                 (mu_a ** 2 + var_a) * cdf +
                 (mu_b ** 2 + var_b) * (1.0 - cdf) - z_mu ** 2)
        if self._keep_variance_fn is not None:
            z_var = self._keep_variance_fn(z_var)
        return z_mu, z_var

    def _max_pool_1x2(self, inputs_mean, inputs_variance):
        mu_a = inputs_mean[:, :, :, 0::2]
        mu_b = inputs_mean[:, :, :, 1::2]
        var_a = inputs_variance[:, :, :, 0::2]
        var_b = inputs_variance[:, :, :, 1::2]
        outputs_mean, outputs_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return outputs_mean, outputs_variance

    def _max_pool_2x1(self, inputs_mean, inputs_variance):
        mu_a = inputs_mean[:, :, 0::2, :]
        mu_b = inputs_mean[:, :, 1::2, :]
        var_a = inputs_variance[:, :, 0::2, :]
        var_b = inputs_variance[:, :, 1::2, :]
        outputs_mean, outputs_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return outputs_mean, outputs_variance

    def forward(self, inputs_mean, inputs_variance) -> list:
        z_mean, z_variance = self._max_pool_1x2(inputs_mean, inputs_variance)
        outputs_mean, outputs_variance = self._max_pool_2x1(z_mean, z_variance)
        return [outputs_mean, outputs_variance]


class Linear(nn.Linear):
    def __init__(self, keep_variance_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._keep_variance_fn = keep_variance_fn

    def forward(self, inputs_mean, inputs_variance) -> list:
        outputs_mean = F.linear(inputs_mean, self.weight, self.bias)
        outputs_variance = F.linear(inputs_variance, self.weight ** 2, None)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return [outputs_mean, outputs_variance]


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, keep_variance_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._keep_variance_fn = keep_variance_fn

    def forward(self, inputs_mean, inputs_variance) -> list:
        self._check_input_dim(inputs_mean)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        outputs_mean = F.batch_norm(
            inputs_mean,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

        outputs_variance = inputs_variance
        weight = ((self.weight.unsqueeze(0)).unsqueeze(2)).unsqueeze(3)
        outputs_variance = outputs_variance * weight ** 2

        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return [outputs_mean, outputs_variance]


class Conv2d(nn.Conv2d):
    def __init__(self, keep_variance_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._keep_variance_fn = keep_variance_fn

    def forward(self, inputs_mean, inputs_variance) -> list:
        if self.padding_mode == 'zeros':
            outputs_mean = F.conv2d(
                inputs_mean, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            outputs_variance = F.conv2d(
                inputs_variance, self.weight ** 2, None, self.stride, self.padding, self.dilation, self.groups)
        else:
            outputs_mean = F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                    self.weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)

        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return [outputs_mean, outputs_variance]


class Upsample(nn.Upsample):
    def __init__(self, keep_variance_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._keep_variance_fn = keep_variance_fn

    def forward(self, inputs_mean, inputs_variance) -> list:
        outputs_mean = super().forward(inputs_mean)
        outputs_variance = super().forward(inputs_variance)

        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return [outputs_mean, outputs_variance]


class Sequential(nn.Sequential):
    # def forward(self, inputs, inputs_variance):
    #     for module in self._modules.values():
    #         inputs, inputs_variance = module(inputs, inputs_variance)
    #
    #     return inputs, inputs_variance
    def forward(self, *input):
        for module in self:
            input = module(*input)
        return input
