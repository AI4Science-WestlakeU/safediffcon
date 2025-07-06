import math
from functools import partial
from typing import Optional

import torch
from torch import jit


def softmin(input: torch.Tensor,
            eps: float,
            dim: int) -> torch.Tensor:
    # `BxNxM` -> `BxN` (if dim=2) or `BxM` (if dim=1)
    return -eps * (-input / eps).logsumexp(dim=dim)


def centerize(c: torch.Tensor,
              x: torch.Tensor,
              y: torch.Tensor) -> torch.Tensor:
    # `BxNxM` -> `BxNxM`
    return c - x.unsqueeze(-1) - y.unsqueeze(1)


@jit.script
def log_sinkhorn(x: torch.Tensor,
                 y: torch.Tensor,
                 a: torch.Tensor,
                 b: torch.Tensor,
                 eps: float,
                 p: float,
                 threshold: float,
                 num_inner_iter,
                 max_iter) -> torch.Tensor:
    """ The stabilized Sinkhorn algorithms of the Sinkhorn's algorithm:
    https://github.com/google-research/google-research/blob/master/soft_sort/sinkhorn.py
    :param x: the input tensor of shape `BxN`
    :param y: the target tensor of shape `BxM`
    :param a: the weight tensor associated to `x`
    :param b: the weight tensor associated to `y`
    :param eps: the value of the entropic regularization parameter
    :param p: the power to be applied to the distance kernel
    :param threshold: the value of sinkhorn error below which to stop iterating
    :param num_inner_iter:
    :param max_iter: the total number of iterations
    :return: transform matrix of shape `BxNxM`
    """

    assert x.size(0) == y.size(0)
    assert x.size() == a.size()
    assert y.size() == b.size()

    cost = (x.unsqueeze(-1) - y.unsqueeze(1)).abs_().pow_(p)  # BxNxM
    log_a = a.log()
    log_b = b.log()
    alpha = torch.zeros_like(log_a)
    beta = torch.zeros_like(log_b)

    for i in range(max_iter):
        alpha += (eps * log_a + softmin(centerize(cost, alpha, beta), eps, 2))
        beta += (eps * log_b + softmin(centerize(cost, alpha, beta), eps, 1))

        if i % num_inner_iter == 0:
            _b = (-centerize(cost, alpha, beta) / eps).exp().sum(dim=1)
            if ((b - _b).abs() / b).sum() < threshold:
                break
    # return transport
    return (-centerize(cost, alpha, beta) / eps).exp()


class SoftQuantizer(object):
    """
    :param input: the input tensor of shape `BxN`
    :param input_weights: the weight tensor associated to `input`
    :param target: the target tensor of shape `BxM`
    :param target_weights: the weight tensor associated to `target`
    :param num_targets: `M`
    :param eps: the value of the entropic regularization parameter
    :param p: the power to be applied to the distance kernel
    :param threshold: the value of sinkhorn error below which to stop iterating
    :param num_inner_iter:
    :param max_iter:
    """

    def __init__(self,
                 input: torch.Tensor,
                 input_weights: Optional[torch.Tensor] = None,
                 target: Optional[torch.Tensor] = None,
                 target_weights: Optional[torch.Tensor] = None,
                 num_targets: Optional[int] = None,
                 eps: float = 1e-3,
                 p: int = 2,
                 threshold: float = 1e-3,
                 rescale_input: bool = False,
                 num_inner_iter: int = 20,
                 max_iter: int = 1_000):

        self.eps = eps
        self.p = p
        self.threshold = threshold
        self._rescale_input = rescale_input
        self._x = None
        self._y = None
        self._a = None
        self._b = None
        self.transport = None
        self.sinkhorn = partial(log_sinkhorn, eps=eps, p=p, threshold=threshold,
                                num_inner_iter=num_inner_iter, max_iter=max_iter)
        self.reset(input, input_weights, target, target_weights, num_targets)

    def reset(self,
              input: torch.Tensor,
              input_weights: Optional[torch.Tensor] = None,
              target: Optional[torch.Tensor] = None,
              target_weights: Optional[torch.Tensor] = None,
              num_targets: Optional[int] = None
              ) -> torch.Tensor:
        assert input.dim() == 2

        n = input.size(1)
        m = input.size(1) if num_targets is None else num_targets

        if input_weights is None:
            # [1/n, ...]
            input_weights = torch.ones_like(input) / n

        if target is None:
            # [0, 1, 2, 3, ..., m - 1] / (m - 1)
            target = torch.linspace(0, 1, m, dtype=input.dtype, device=input.device)
            target = target.repeat(input.size(0), m)  # BxM

        if target_weights is None:
            # [1/m, ...]
            target_weights = torch.ones_like(target) / m

        self._x = self.rescale_input(input) if self._rescale_input else input
        self._y = target
        self._a = input_weights
        self._b = target_weights

        self.transport = self.sinkhorn(x=self._x, y=self._y, a=self._a, b=self._b)
        return self.transport

    @property
    def softcdf(self) -> torch.Tensor:
        return (1 / self._a) * torch.einsum("bnm,bm->bn", self.transport, self._b.cumsum(dim=1))

    @property
    def softsort(self) -> torch.Tensor:
        return (1 / self._b) * torch.einsum("bnm,bn->bm", self.transport, self._x)

    @staticmethod
    def rescale_input(input: torch.Tensor,
                      scale: float = 1.0,
                      min_std: float = 1e-10,
                      is_logistic=True) -> torch.Tensor:
        mean = input.mean(dim=1, keepdim=True)
        std = input.std(dim=1, keepdim=True).clamp_min_(min_std)
        f = torch.atan
        if is_logistic:
            scale *= math.sqrt(3) / math.pi
            f = torch.sigmoid
        std *= scale
        return f((input - mean) / std)
