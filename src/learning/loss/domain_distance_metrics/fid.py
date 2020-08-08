import torch


from path_learning.utils import get_logger
from path_learning.utils.torch_utils import torch_cov, torch_symsqrt


logger = get_logger("frechet_inception_distance")


def frechet_inception_distance(data_domain_1: torch.Tensor, data_domain_2: torch.Tensor, eps=1e-6) -> torch.Tensor:
    # Fréchet Inception Distance (FID) for domain-confusion loss
    # Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016).
    # Improved techniques for training gans. In Advances in Neural Information
    # Processing Systems (pp. 2234–2242).

    """Torch implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Torch tensor containing the mean activations of domain 1
    -- mu2   : Torch tensor containing the mean activations of domain 2
    -- sigma1: Torch tensor containing the covariance matrix of domain 1
    -- sigma2: Torch tensor containing the covariance matrix of domain 2
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = torch.mean(data_domain_1, dim=0)
    mu2 = torch.mean(data_domain_2, dim=0)
    sigma1 = torch_cov(data_domain_1, rowvar=False)
    sigma2 = torch_cov(data_domain_2, rowvar=False)

    assert len(mu1.size()) >= 1
    assert len(mu2.size()) >= 1
    assert len(sigma1.size()) >= 2
    assert len(sigma1.size()) >= 2

    assert mu1.size() == mu2.size(), \
        'Training and test mean vectors have different lengths'
    assert sigma1.size() == sigma2.size(), \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    sigma_mm = torch.mm(sigma1, sigma2)

    try:
        covmean = torch_symsqrt(sigma_mm)

        if not torch.all(torch.isfinite(covmean)):
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            logger.warning(msg)
            offset = torch.eye(sigma1.size(0)) * eps
            covmean = torch_symsqrt(torch.mm((sigma1 + offset), (sigma2 + offset)))

        out = torch.dot(diff, diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)

        return out
    except RuntimeError as exception:
        # it might occur that symsqrt is not converging
        # RuntimeError: svd_cuda: the updating process of SBDSDC did not converge (error: 9)
        logger.error(exception)

        # instead we just return an approximation of the fréchet distance
        out = torch.dot(diff, diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(torch.sqrt(sigma_mm))
        return out