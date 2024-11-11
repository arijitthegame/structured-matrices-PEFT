from typing import Optional, List, Dict
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from lora.circulant_lora import CirculantLayer
from lora.kronecker_lora import KroneckerLayer
from lora.toeplitz_lora import ToeplitzLayer

ACT2CLS = {
    "gelu": nn.GELU,
    "linear": nn.Identity,
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


################################################################################
## Simple helper functions to save and train specific modules
# TODO: Make them modular and combine them in one function


def mark_only_circulant_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """
    Helper function to make sure only the circulant parameters are trained
    """
    for n, p in model.named_parameters():
        if "circulant_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "circulant_only":
        for m in model.modules():
            if (
                isinstance(m, CirculantLayer)
                and hasattr(m, "bias")
                and m.bias is not None
            ):
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def circulant_state_dict(
    model: nn.Module, bias: str = "none"
) -> Dict[str, torch.Tensor]:
    """
    Helper function to extarct only the trainable circulant parameters
    """
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "circulant_" in k.lower()}
    elif bias == "all":
        return {
            k: my_state_dict[k]
            for k in my_state_dict
            if "circulant_" in k.lower() or "bias" in k
        }
    elif bias == "circulant_only":
        to_return = {}
        for k in my_state_dict:
            if "circulant_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("circulant_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def mark_only_kronecker_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Helper function to make sure only the Kronecker parameters are optimized"""
    for n, p in model.named_parameters():
        if "kronecker_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "kronecker_only":
        for m in model.modules():
            if (
                isinstance(m, KroneckerLayer)
                and hasattr(m, "bias")
                and m.bias is not None
            ):
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def kronecker_state_dict(
    model: nn.Module, bias: str = "none"
) -> Dict[str, torch.Tensor]:
    """Helper function to save only the optimized Kronecker weights"""
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "kronecker_" in k.lower()}
    elif bias == "all":
        return {
            k: my_state_dict[k]
            for k in my_state_dict
            if "kronecker_" in k.lower() or "bias" in k
        }
    elif bias == "kronecker_only":
        to_return = {}
        for k in my_state_dict:
            if "kronecker_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("kronecker_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def mark_only_toeplitz_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """ "Helper function to make sure that only the Toeplitz parameters are optimized"""
    for n, p in model.named_parameters():
        if "toeplitz_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "toeplitz_only":
        for m in model.modules():
            if (
                isinstance(m, ToeplitzLayer)
                and hasattr(m, "bias")
                and m.bias is not None
            ):
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def toeplitz_state_dict(
    model: nn.Module, bias: str = "none"
) -> Dict[str, torch.Tensor]:
    """Function to save the Toeplitz parameters"""
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "toeplitz_" in k.lower()}
    elif bias == "all":
        return {
            k: my_state_dict[k]
            for k in my_state_dict
            if "toeplitz_" in k.lower() or "bias" in k
        }
    elif bias == "toeplitz_only":
        to_return = {}
        for k in my_state_dict:
            if "toeplitz_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("toeplitz_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


##########################################################################################################

## Functions to handle LDRMs and Kronecker products

# helper functions for Toeplitz matrices


def toeplitz(c, r):
    """Create a Toeplitz matrix from 1st row and column"""
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j - i].reshape(*shape)


def toeplitz_multiplication(c, r, A, **kwargs):
    """
    Compute A @ scipy.linalg.toeplitz(c,r) using the FFT.

    Parameters
    ----------
    c: ndarray
        the first column of the Toeplitz matrix
    r: ndarray
        the first row of the Toeplitz matrix
    A: ndarray
        the matrix to multiply on the left
    """

    m = c.shape[0]
    n = r.shape[0]

    fft_len = int(2 ** np.ceil(np.log2(m + n - 1)))
    zp = fft_len - m - n + 1

    if A.shape[-1] != n:
        raise ValueError("A dimensions not compatible with toeplitz(c,r)")

    x = torch.cat((r, torch.zeros(zp, dtype=c.dtype, device=r.device), c.flip(0)[:-1]))
    xf = torch.fft.rfft(x, n=fft_len)

    Af = torch.fft.rfft(A, n=fft_len, dim=-1)

    return torch.fft.irfft((Af * xf).T, n=fft_len, dim=0)[
        :n,
    ].T


# util function that constructs a batched Kronecker product
def batch_kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


# helper functions for circulant matrices
def circulant_multiply(c, x):
    """Left Multiply circulant matrix with first column c by x.
    E.g. if the matrix is
        [1 2 3]
        [2 3 1]
        [3 1 2]
    c should be [1,2,3]
    Parameters:
        c: (n, )
        x: (batch_size, .., n) or (n, )
    Return:
        prod: (batch_size, ..., n) or (n, )
    """
    return torch.fft.irfft(torch.fft.rfft(x) * torch.fft.rfft(c))


def circulant(tensor, dim=0):
    """get a circulant version of the tensor along the {dim} dimension.

    The additional axis is appended as the last dimension.
    E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
    S = tensor.shape[dim]
    tmp = torch.cat(
        [
            tensor.flip((dim,)),
            torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S - 1),
        ],
        dim=dim,
    )
    return tmp.unfold(dim, S, 1).flip((-1,))


def skew_mult(r, v, root_vec):
    """
    Computes the fast matrix vector multiplication by a skew circulant matrix.
    r = vec of size hidden dim (1st row)
    v = N-d tensor
    root_vec := [1, \eta, ...,\eta^k], where \eta is the primitive complex root of x^n - f
    """

    n = v.size(-1)
    r_skew = r * root_vec
    r_fft = torch.fft.fft(r_skew)
    v_skew = v * root_vec
    v_fft = torch.fft.fft(v_skew)
    return (torch.conj(root_vec) * torch.fft.ifft(r_fft * v_fft)).real


def skew_circulant_matrix(row):
    """Constructs a skew circulant matrix from first row"""
    n = row.numel()
    skew_circulant_matrix = row.unsqueeze(0)

    for i in range(1, n):
        rolled = torch.roll(row, shifts=i, dims=0)
        rolled[:i] *= -1
        skew_circulant_matrix = torch.cat(
            (skew_circulant_matrix, rolled.unsqueeze(0)), dim=0
        )

    return skew_circulant_matrix


def compute_root(n):
    """Computes root of x^n + 1"""
    coeff = np.zeros(n + 1)
    coeff[0] = 1
    coeff[-1] = 1
    e = np.roots(coeff)[0]
    return e
