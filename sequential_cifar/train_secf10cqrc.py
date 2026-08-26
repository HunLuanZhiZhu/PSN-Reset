import math
from typing import Tuple
import torch
from torch import Tensor
torch._dynamo.config.recompile_limit = 1024
import torch.nn as nn
import torch.nn.functional as F
_seed_ = 2020
import random
random.seed(2020)

import torchvision
from torchvision import transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
import glif


@torch.jit.script
def round_to_pow2(x: torch.Tensor):
    sign = torch.sign(x)
    p = torch.log2(x.abs())
    p_round = torch.round(p)
    return sign * torch.pow(2., p_round)


class RoundToPow2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return round_to_pow2(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output

class ForwardFirstBackwardSecond(torch.autograd.Function):
    @staticmethod
    def forward(_, input1: Tensor, input2: Tensor) -> Tensor:
        return input1

    @staticmethod
    def backward(_, *grad_output: Tensor):
        grad_output, = grad_output
        return None, grad_output

@torch.jit.script
def floor_to_pow2(x: torch.Tensor)-> torch.Tensor:
    return torch.where(x > 0, x.log2().floor_().exp2_(), 0)

@torch.jit.script
def spiking(x: torch.Tensor)-> torch.Tensor:
    return ForwardFirstBackwardSecond.apply(
        floor_to_pow2(x),
        F.gelu(x,approximate='tanh')
    )

def to_c_last(x_seq: torch.Tensor):
    if x_seq.dim() == 3:
        return x_seq
    else:
        return x_seq.transpose(2, -1)

class BN1d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(num_features, **factory_kwargs))
        self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
        self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))

    def get_mean_var(self, input: Tensor=None) -> Tuple[Tensor, Tensor]:
        if self.training:
            mean = input.mean([0, 2])
            var = input.var([0, 2], unbiased=False)
            with torch.no_grad():
                self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1. - self.momentum) * self.running_var + self.momentum * input.var([0, 2], unbiased=True)
        else:
            mean, var = self.running_mean, self.running_var

        return mean, var


@torch.jit.script
def reset(h_seq: torch.Tensor, betas: torch.Tensor, inv_betas: torch.Tensor):
    with torch.no_grad():
         us = floor_to_pow2(h_seq[:-2]).mul(betas[:-2]).cumsum_(dim=0).mul_(inv_betas[1:-1])
         h_seq1 = h_seq[:-1].detach().clone()
         h_seq1[1:] -= us
         us = floor_to_pow2(h_seq1).mul(betas[:-1]).cumsum_(dim=0).mul_(inv_betas[1:])
    h_seq[1:] -= us


class ResetModule(nn.Module):
    def __init__(self, C: int, K: int):
        super().__init__()
        self.C = C
        self.K = K-1
        self.conv = nn.Conv1d(C, C, kernel_size=self.K, groups=C, bias=False)
        nn.init.uniform_(self.conv.weight, -1e-10, 1e-10)

    def forward(self, h_seq: torch.Tensor):
        h_seq = h_seq.detach()
        spike = floor_to_pow2(h_seq)
        spike_pad = F.pad(spike[..., :-1], (self.K, 0))
        us = F.conv1d(spike_pad, self.conv.weight, stride=1, padding=0, groups=self.C)
        return us


class IFNode5(nn.Module):
    """Complex temporal filter: each complex channel has a complex kernel (k_re + i*k_im)."""
    def __init__(self, T: int, C: int, surrogate_function =  None, use_reset: bool = True):
        super().__init__()

        self.T = T
        self.C = C
        self.K = T
        self.Ch = C // 2
        self.surrogate_function = spiking

        self.conv_re = nn.Conv1d(self.Ch, self.Ch, kernel_size=T, groups=self.Ch, bias=False)
        self.conv_im = nn.Conv1d(self.Ch, self.Ch, kernel_size=T, groups=self.Ch, bias=False)

        self.bn1d = BN1d(C)
        nn.init.constant_(self.bn1d.bias, -0)

        if use_reset:
            self.reset_module = ResetModule(C, T)

    def forward(self, x_seq: torch.Tensor):
        shape = x_seq.shape
        T, N, C = shape[:3]
        Ch = C // 2
        eps = self.bn1d.eps

        x_seq = x_seq.reshape(T, N, C, -1)
        E = x_seq.shape[-1]

        x_seq = x_seq.permute(1, 3, 2, 0).reshape(-1, C, T)

        x_re = x_seq[:, :Ch, :]
        x_im = x_seq[:, Ch:, :]

        x_re = F.pad(x_re, (self.K - 1, 0))
        x_im = F.pad(x_im, (self.K - 1, 0))

        if self.training:
            t_re = self.conv_re(x_re) - self.conv_im(x_im)
            t_im = self.conv_re(x_im) + self.conv_im(x_re)
            t = torch.cat([t_re, t_im], dim=1)
            mean, var = self.bn1d.get_mean_var(t)
        else:
            mean, var = self.bn1d.get_mean_var()

        gamma_re = self.bn1d.weight[:Ch]
        gamma_im = self.bn1d.weight[Ch:]
        var_re = var[:Ch]
        var_im = var[Ch:]
        mean_re = mean[:Ch]
        mean_im = mean[Ch:]

        weight_re = self.conv_re.weight
        weight_im = self.conv_im.weight

        fused_rr = (
            weight_re.transpose(0, 2)
            * gamma_re
            / (var_re + eps).sqrt()
        ).transpose(0, 2)
        fused_rr = RoundToPow2.apply(fused_rr)

        fused_ir = (
            weight_im.transpose(0, 2)
            * gamma_re
            / (var_re + eps).sqrt()
        ).transpose(0, 2)
        fused_ir = RoundToPow2.apply(fused_ir)

        fused_ri = (
            weight_re.transpose(0, 2)
            * gamma_im
            / (var_im + eps).sqrt()
        ).transpose(0, 2)
        fused_ri = RoundToPow2.apply(fused_ri)

        fused_ii = (
            weight_im.transpose(0, 2)
            * gamma_im
            / (var_im + eps).sqrt()
        ).transpose(0, 2)
        fused_ii = RoundToPow2.apply(fused_ii)

        bias_re = self.bn1d.bias[:Ch] - mean_re * gamma_re / (var_re + eps).sqrt()
        bias_im = self.bn1d.bias[Ch:] - mean_im * gamma_im / (var_im + eps).sqrt()

        h_re = (
            F.conv1d(x_re, fused_rr, bias=bias_re, groups=self.Ch)
            - F.conv1d(x_im, fused_ir, groups=self.Ch)
        )
        h_im = (
            F.conv1d(x_re, fused_ii, bias=bias_im, groups=self.Ch)
            + F.conv1d(x_im, fused_ri, groups=self.Ch)
        )
        h_seq = torch.cat([h_re, h_im], dim=1)

        h_seq1 = h_seq
        if hasattr(self, 'reset_module'):
            for _ in range(3):
                us = self.reset_module(h_seq1)
                h_seq1 = h_seq - us

        h_seq = h_seq1.view(N, E, C, T).permute(3, 0, 2, 1).contiguous()

        spike = self.surrogate_function(h_seq)

        return spike.view(shape)


class MaskedSlidingPSN(nn.Module):
    def gen_gemm_weight(self, T: int):
        weight = torch.zeros([T, T], device=self.weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.order)
            length = min(end - start, self.order)
            weight[i][start: end] = self.weight[self.order - length: self.order]

        return weight


    def __init__(self, order: int, surrogate_function, exp_init: bool, backend='gemm'):
        super().__init__()
        self.order = order
        self.backend = backend
        if self.backend == 'gemm':
            if exp_init:
                weight = torch.ones([order])
                for i in range(order - 2, -1, -1):
                    weight[i] = weight[i + 1] / 2.

                self.weight = nn.Parameter(weight)
            else:
                self.weight = torch.ones([1, order])
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                self.weight = nn.Parameter(self.weight[0])

            self.threshold = nn.Parameter(torch.as_tensor(-1.))
            self.surrogate_function = surrogate_function

        elif self.backend == 'conv':
            self.weight = torch.zeros([1, 1, order])
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.weight = nn.Parameter(self.weight)
            self.threshold = nn.Parameter(torch.as_tensor(1.))
            self.surrogate_function = surrogate_function

        self.betas : nn.Tensor
        self.inv_betas : nn.Tensor
        T = 32
        arange_tensor = torch.arange(T-1, -1, -1, dtype=torch.float64)
        betas = (2**-1) ** arange_tensor
        betas = betas.unsqueeze(-1).float()
        inv_betas = betas.reciprocal()
        self.register_buffer('betas', betas[:-1])
        self.register_buffer('inv_betas', inv_betas[1:])

    def forward(self, x_seq: torch.Tensor):
        if self.backend == 'gemm':
            weight = self.gen_gemm_weight(x_seq.shape[0])
            h_seq = torch.addmm(self.threshold, weight, x_seq.flatten(1))

            reset(h_seq, self.betas, self.inv_betas)

            return self.surrogate_function(h_seq.view_as(x_seq))

        elif self.backend == 'conv':
            x_seq_shape = x_seq.shape
            x_seq = x_seq.flatten(1).t().unsqueeze(1)
            x_seq = F.pad(x_seq, pad=(self.order - 1, 0))
            x_seq = F.conv1d(x_seq, self.weight, stride=1)
            x_seq = x_seq.squeeze(1).t().view(x_seq_shape)

            x_seq -= self.threshold
            h_seq = x_seq

            reset(h_seq, self.betas, self.inv_betas)

            return self.surrogate_function(h_seq)
        else:
            raise NotImplementedError(self.backend)


class IFNode5PorderMaskD(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase, P: int):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.fc = DecayPorderMaskedLinear(P, T, T)
        nn.init.constant_(self.fc.bias, -1)

        self.betas : nn.Tensor
        self.inv_betas : nn.Tensor
        arange_tensor = torch.arange(T-1, -1, -1, dtype=torch.float64)
        betas = (2**-1) ** arange_tensor
        betas = betas.unsqueeze(-1).float()
        inv_betas = betas.reciprocal()
        self.register_buffer('betas', betas[:-1])
        self.register_buffer('inv_betas', inv_betas[1:])

    def forward(self, x_seq: torch.Tensor):
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.masked_weight(), x_seq.flatten(1))

        reset(h_seq, self.betas, self.inv_betas)

        spike = self.surrogate_function(h_seq)
        return spike.view(x_seq.shape)


class LIFNR(nn.Module):
    def __init__(self, tau: float, surrogate_function: surrogate.SurrogateFunctionBase):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.v_th = 1.
        self.tau = tau
        T= 32
        self.betas : nn.Tensor
        self.inv_betas : nn.Tensor
        arange_tensor = torch.arange(T-1, -1, -1, dtype=torch.float64)
        betas = (1.-1./tau) ** arange_tensor
        betas = betas.unsqueeze(-1).float()
        inv_betas = betas.reciprocal()
        self.register_buffer('betas', betas[:-1])
        self.register_buffer('inv_betas', inv_betas[1:])

    @staticmethod
    @torch.jit.script
    def pre_forward(x_seq: torch.Tensor, tau: float):
        decay_a = 1. - 1. / tau
        decay_b = 1. / tau
        h_t = torch.zeros_like(x_seq[0])
        h_seq = []
        for t in range(x_seq.shape[0]):
            h_t = decay_a * h_t + decay_b * x_seq[t]
            h_seq.append(h_t)

        return torch.stack(h_seq)

    def forward(self, x_seq: torch.Tensor):
        h_seq = self.pre_forward(x_seq, self.tau)

        h_seq -= self.v_th
        h_seq_flat = h_seq.flatten(1)
        reset(h_seq_flat, self.betas, self.inv_betas)
        h_seq = h_seq_flat.view(h_seq.shape)

        spike = self.surrogate_function(h_seq)
        return spike

class DecayPorderMaskedLinear(nn.Linear):
    def __init__(self, P: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P = P
        mask1 = torch.ones_like(self.weight.data)
        mask0 = torch.tril(mask1) * torch.triu(mask1, -(P - 1))
        self.register_buffer('mask0', mask0)
        self.register_buffer('mask1', mask1)
        self.k = 0.

    @staticmethod
    @torch.jit.script
    def gen_mask(k: float, mask0: torch.Tensor, mask1: torch.Tensor):
        return k * mask0 + (1. - k) * mask1

    @staticmethod
    @torch.jit.script
    def gen_masked_weight(weight: torch.Tensor, k: float, mask0: torch.Tensor, mask1: torch.Tensor):
        return weight * (k * mask0 + (1. - k) * mask1)

    def masked_weight(self):
        return self.gen_masked_weight(self.weight, self.k, self.mask0, self.mask1)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight * self.gen_mask(self.k, self.mask0, self.mask1), self.bias)


class FunctionalLayer(nn.Module):
    def __init__(self, f):
        super(FunctionalLayer, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


def unsqueeze_last_dim(x):
    return x.unsqueeze(-1)


def squeeze_last_dim(x):
    return x.squeeze(-1)


def unsqueeze_first_two_dims(x):
    return x.unsqueeze(-1).unsqueeze(-1)


def squeeze_first_two_dims(x):
    return x.squeeze(-1).squeeze(-1)


class MixupCollate:
    def __init__(self, mixup_transform):
        self.mixup_transform = mixup_transform

    def __call__(self, batch):
        return self.mixup_transform(*default_collate(batch))


def create_neuron(neu: str, **kwargs):
    if neu == 'mspsn':
        return MaskedSlidingPSN(order=kwargs['P'], surrogate_function=kwargs['surrogate_function'], exp_init=kwargs['exp_init'], backend='gemm')

    elif neu == 'lifnr':
        return LIFNR(tau=2., surrogate_function=kwargs['surrogate_function'])

    elif neu == 'lif':
        return neuron.LIFNode(tau=2., detach_reset=True, surrogate_function=kwargs['surrogate_function'], v_reset=None, step_mode='m', backend='cupy')

    elif neu == 'plif':
        return neuron.ParametricLIFNode(init_tau=2., detach_reset=True, surrogate_function=kwargs['surrogate_function'], v_reset=None, step_mode='m', backend='cupy')
    elif neu == 'plif_torch':
        return neuron.ParametricLIFNode(init_tau=2., detach_reset=True, surrogate_function=kwargs['surrogate_function'], v_reset=None, step_mode='m', backend='torch')

    elif neu == 'klif':
        return neuron.KLIFNode(surrogate_function=kwargs['surrogate_function'], v_reset=0., step_mode='m')
    elif neu == 'glif':
        if 'channels' in kwargs.keys():
            return nn.Sequential(
                FunctionalLayer(f=unsqueeze_last_dim),
                glif.LIFSpike_CW(kwargs['channels'], **glif.get_initial_dict(kwargs['T'])),
                FunctionalLayer(f=squeeze_last_dim),
            )

        elif 'features' in kwargs.keys():
            return nn.Sequential(
                FunctionalLayer(f=unsqueeze_first_two_dims),
                glif.LIFSpike_CW(kwargs['features'], **glif.get_initial_dict(kwargs['T'])),
                FunctionalLayer(f=squeeze_first_two_dims),
            )
        else:
            raise NotImplementedError

    elif neu == 'if5':
        return IFNode5(T=kwargs['T'], C=kwargs['channels'], surrogate_function=kwargs['surrogate_function'], use_reset=kwargs.get('use_reset', False))


    if neu == 'if5pmd8':
        return IFNode5PorderMaskD(T=kwargs['T'], surrogate_function=kwargs['surrogate_function'], P=kwargs['P'])


class ClassificationPresetTrain:
    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = []
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

from torch import Tensor
from typing import Tuple
class RandomMixup(torch.nn.Module):
    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

class RandomCutmix(torch.nn.Module):
    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = torchvision.transforms.functional.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target


class ComplexConv1d(nn.Module):
    """True complex 1D convolution: (a+bi)*(c+di) = (ac-bd)+(ad+bc)i"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        if x.dim() == 4:
            T, N, _, E = x.shape
            x = x.flatten(0, 1)
            C = x.shape[1] // 2
            x_re, x_im = x[:, :C, :], x[:, C:, :]
            out_re = self.conv_re(x_re) - self.conv_im(x_im)
            out_im = self.conv_re(x_im) + self.conv_im(x_re)
            out = torch.cat([out_re, out_im], dim=1)
            return out.view(T, N, -1, E)
        else:
            C = x.shape[1] // 2
            x_re, x_im = x[:, :C, :], x[:, C:, :]
            out_re = self.conv_re(x_re) - self.conv_im(x_im)
            out_im = self.conv_re(x_im) + self.conv_im(x_re)
            return torch.cat([out_re, out_im], dim=1)


class ComplexBN1d(nn.Module):
    """Complex BN: independent re/im normalization"""
    def __init__(self, num_features):
        super().__init__()
        self.bn_re = nn.BatchNorm1d(num_features)
        self.bn_im = nn.BatchNorm1d(num_features)

    def forward(self, x):
        if x.dim() == 4:
            T, N, _, E = x.shape
            x = x.flatten(0, 1)
            C = x.shape[1] // 2
            x_re = self.bn_re(x[:, :C, :])
            x_im = self.bn_im(x[:, C:, :])
            out = torch.cat([x_re, x_im], dim=1)
            return out.view(T, N, -1, E)
        else:
            C = x.shape[1] // 2
            x_re = self.bn_re(x[:, :C, :])
            x_im = self.bn_im(x[:, C:, :])
            return torch.cat([x_re, x_im], dim=1)


class ComplexLinear(nn.Module):
    """True complex linear: (a+bi)*(c+di) = (ac-bd)+(ad+bc)i"""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear_re = nn.Linear(in_features, out_features, bias=bias)
        self.linear_im = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        if x.dim() == 3:
            T, N, _ = x.shape
            x = x.flatten(0, 1)
            C = x.shape[-1] // 2
            x_re, x_im = x[..., :C], x[..., C:]
            out_re = self.linear_re(x_re) - self.linear_im(x_im)
            out_im = self.linear_re(x_im) + self.linear_im(x_re)
            out = torch.cat([out_re, out_im], dim=-1)
            return out.view(T, N, -1)
        else:
            C = x.shape[-1] // 2
            x_re, x_im = x[..., :C], x[..., C:]
            out_re = self.linear_re(x_re) - self.linear_im(x_im)
            out_im = self.linear_re(x_im) + self.linear_im(x_re)
            return torch.cat([out_re, out_im], dim=-1)


class CIFAR10Net(nn.Module):
    """
    Complex spiking SNN with true complex arithmetic.
    ComplexConv1d/ComplexLinear perform (a+bi)*(c+di) = (ac-bd)+(ad+bc)i.
    First C channels = real part, last C channels = imaginary part.
    floor_to_pow2 applied element-wise as spike function.
    """
    def __init__(self, channels, neu: str, T: int, class_num: int, P:int=-1, exp_init:bool=False, use_reset:bool=False):
        super().__init__()
        C = channels * 2
        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    conv.append(layer.Conv1d(3, C, kernel_size=3, padding=1, bias=False))
                    conv.append(layer.BatchNorm1d(C))
                else:
                    conv.append(ComplexConv1d(channels, channels, kernel_size=3, padding=1, bias=False))
                    conv.append(ComplexBN1d(channels))
                conv.append(create_neuron(neu, T=T, features=C, surrogate_function=surrogate.ATan(), channels=C, P=P, exp_init=exp_init, use_reset=use_reset))

            conv.append(layer.AvgPool1d(2))

        self.conv = nn.Sequential(*conv)

        self.fc = nn.Sequential(
            layer.Flatten(),
            ComplexLinear(channels * 8, channels * 2),
            create_neuron(neu, T=T, features=channels * 4, surrogate_function=surrogate.ATan(), channels=channels * 4, P=P, exp_init=exp_init, use_reset=use_reset),
            layer.Linear(channels * 4, class_num),
        )

        functional.set_step_mode(self, 'm')

    def forward(self, x_seq: torch.Tensor):
        x_seq = x_seq.permute(3, 0, 1, 2)
        x_seq = self.fc(self.conv(x_seq))
        return x_seq.mean(0)




def main():

    parser = argparse.ArgumentParser(description='Classify Sequential CIFAR10/100')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of CIFAR10/100 dataset')
    parser.add_argument('-out-dir', type=str, default='./logscqrc', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-neu', type=str, help='use which neuron')
    parser.add_argument('-class-num', type=int, default=10)

    parser.add_argument('-P', type=int, default=None, help='the order of the masked/sliding PSN')
    parser.add_argument('-exp-init', action='store_true', help='use the exp init method to initialize the weight of SPSN')
    parser.add_argument('-r', action='store_true', help='use reset module in IFNode5')


    args = parser.parse_args()
    print(args)

    mixup_transforms = []
    mixup_transforms.append(RandomMixup(args.class_num, p=1.0, alpha=0.2))
    mixup_transforms.append(RandomCutmix(args.class_num, p=1.0, alpha=1.))
    mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
    collate_fn = MixupCollate(mixupcutmix)

    if args.class_num == 10:
        transform_train = ClassificationPresetTrain(mean=(0.4914, 0.4822, 0.4465),
                                                      std=(0.2023, 0.1994, 0.2010), interpolation=InterpolationMode('bilinear'),
                                                      auto_augment_policy='ta_wide',
                                                      random_erase_prob=0.1)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.class_num == 100:
        transform_train = ClassificationPresetTrain(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                    std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                                                    interpolation=InterpolationMode('bilinear'),
                                                    auto_augment_policy='ta_wide',
                                                    random_erase_prob=0.1)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])

    else:
        raise NotImplementedError(args.class_num)



    if args.class_num == 10:
        train_set = torchvision.datasets.CIFAR10(
                root=args.data_dir,
                train=True,
                transform=transform_train,
                download=True)

        test_set = torchvision.datasets.CIFAR10(
                root=args.data_dir,
                train=False,
                transform=transform_test,
                download=True)

    elif args.class_num == 100:
        train_set = torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=True,
            transform=transform_train,
            download=True)

        test_set = torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=False,
            transform=transform_test,
            download=True)
    else:
        raise NotImplementedError(args.class_num)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True,
        persistent_workers=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True,
        persistent_workers=True
    )
    out_dir = f'{args.neu}_e{args.epochs}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}'
    if args.amp:
        out_dir += '_amp'


    if args.P is not None:
        out_dir += f'_P{args.P}'
        if args.exp_init:
            out_dir += '_ei'
    if args.r:
        out_dir += '_r'

    pt_dir = os.path.join(args.out_dir+str(args.class_num), 'pt', out_dir)
    out_dir = os.path.join(args.out_dir+str(args.class_num), out_dir)

    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)

    net = CIFAR10Net(channels=args.channels, neu=args.neu, T=32, class_num=args.class_num, P=args.P, exp_init=args.exp_init, use_reset=args.r)
    net.to(args.device)
    net = torch.compile(net)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        print(max_test_acc)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))



    for epoch in range(start_epoch, args.epochs):
        if args.neu == 'if5pmd8':
            for m in net.modules():
                if isinstance(m, DecayPorderMaskedLinear):
                    mk = epoch / (args.epochs - 1)
                    m.k = min(mk * 8, 1.)


        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        for batch_index, (img, label) in enumerate(train_data_loader):
            optimizer.zero_grad()
            img = img.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                y = net(img)
                loss = F.cross_entropy(y, label, label_smoothing=0.1)


            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_samples += label.shape[0]
            train_loss += loss.item() * label.shape[0]
            train_acc += (y.argmax(1) == label.argmax(1)).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                y = net(img)
                loss = F.cross_entropy(y, label)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (y.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()
