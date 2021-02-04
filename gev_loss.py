import numpy as np
import scipy.special

import torch


_EPSILON = 1e-6


def _check_label(target):
    if not torch.all((target == 1) | (target == 0)):
        raise ValueError("Labels must be binary (1 or 0)")


class ExponentialIntegral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        x = input.detach().numpy()
        return torch.from_numpy(scipy.special.expi(x))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.exp(input) / input
        return grad_input * grad_output


expint = ExponentialIntegral.apply


def _gammainc(x, xi):
    if xi > 0:
        return np.where(x != 0, scipy.special.gamma(xi) * scipy.special.gammaincc(xi, x), scipy.special.gamma(xi))
    elif xi == 0:
        return -scipy.special.expi(-x)
    else:
        return -x ** xi * np.exp(-x) / xi + _gammainc(x, xi + 1) / xi


class GammaIncomplete(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, xi):
        ctx.save_for_backward(input)
        ctx.xi = xi
        x = input.detach().numpy()
        return torch.from_numpy(_gammainc(x, xi))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = -input ** (ctx.xi - 1) * torch.exp(-input)
        grad = grad_input * grad_output
        return grad, None


gammainc = GammaIncomplete.apply


def gev_entropy(p: torch.Tensor, xi: float) -> torch.Tensor:
    assert torch.all(0 <= p) and torch.all(p <= 1)

    logp = torch.clamp(p, _EPSILON, 1 - _EPSILON).log()
    if xi == 0:
        return -p * torch.log(-logp) + expint(logp)
    else:
        return 1 / xi * (gammainc(-logp, 1 - xi) - p)


def gev_entropy_dual(x: torch.Tensor, xi: float) -> torch.Tensor:
    if xi == 0:
        return -expint(-torch.exp(-x))
    elif xi > 0:
        p = torch.clamp(gev_cdf(x, xi), _EPSILON, 1 - _EPSILON)
        return torch.where(x > -1 / xi, gammainc(-p.log(), -xi), torch.zeros_like(x))
    else:
        p = torch.clamp(gev_cdf(x, xi), _EPSILON, 1 - _EPSILON)
        return torch.where(x < -1 / xi, gammainc(-p.log(), -xi), x + scipy.special.gamma(-xi) + 1 / xi)


def gev_cdf(x: torch.Tensor, xi: float, eps: float = _EPSILON) -> torch.Tensor:
    if xi == 0:
        return torch.exp(-torch.exp(-x))
    else:
        return torch.exp(-torch.clamp(1 + xi * x, min=eps) ** (-1 / xi))


def gev_inverse_link(x: torch.Tensor, xi: float, eps: float = _EPSILON) -> torch.Tensor:
    prob = gev_cdf(x, xi, eps)
    y_proba = torch.stack((1 - prob, prob), 1)
    return y_proba


class GEVFenchelYoungLoss(torch.nn.Module):
    def __init__(self, xi: float):
        super(GEVFenchelYoungLoss, self).__init__()
        self.xi = xi

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _check_label(target)

        xi = self.xi
        l = gev_entropy_dual(input, xi) + gev_entropy(target, xi) - target * input

        return l.mean()


class GEVCanonicalLoss(torch.nn.Module):
    def __init__(self, xi: float):
        super(GEVCanonicalLoss, self).__init__()
        self.xi = xi

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _check_label(target)

        xi = self.xi

        if xi == 0:
            l = gammainc(torch.exp(-input), xi) - target * input
        elif xi > 0:
            p = torch.clamp(gev_cdf(input, xi), _EPSILON, 1 - _EPSILON)
            l = gammainc(-p.log(), -xi) - target * input.clamp(min=-1 / xi)
        else:
            p = torch.clamp(gev_cdf(input, xi), _EPSILON, 1 - _EPSILON)
            l = gammainc(-p.log(), -xi) - target * input.clamp(max=-1 / xi)
        l += gev_entropy(target, xi)

        return l.mean()


class GEVLogisticLoss(torch.nn.Module):
    def __init__(self, xi: float):
        super(GEVLogisticLoss, self).__init__()
        self.xi = xi

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _check_label(target)

        p = gev_cdf(input, self.xi, eps=1e-2)
        l = -target * p.log() - (1 - target) * (1 - p).log()
        return l.mean()
