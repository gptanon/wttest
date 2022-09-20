# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import torch


class TorchBackend:
    """
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    backend\torch_backend.py

    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    name = 'torch'

    @classmethod
    def input_checks(cls, x):
        if x is None:
            raise TypeError('The input should be not empty.')

    @classmethod
    def complex_check(cls, x):
        if not cls._is_complex(x):
            raise TypeError('The input should be complex (got %s).' % x.dtype)

    @classmethod
    def real_check(cls, x):
        if not cls._is_real(x):
            raise TypeError('The input should be real and float (got %s).' %
                            x.dtype)

    @staticmethod
    def contiguous_check(x):
        if not x.is_contiguous():
            raise RuntimeError('Tensors must be contiguous.')

    @staticmethod
    def _is_complex(x):
        return torch.is_complex(x)

    @staticmethod
    def _is_real(x):
        return 'float' in str(x.dtype)

    @classmethod
    def modulus(cls, x):
        """Complex modulus, i.e. absolute value.

        This method exists in case of future optimizations.
        """
        return torch.abs(x)

    @classmethod
    def cdgmm(cls, A, B):
        """Complex pointwise multiplication.

        This method exists in case of future optimizations.
        """
        return A * B

    @staticmethod
    def concatenate(arrays, axis=-2):
        return torch.stack(arrays, dim=axis)

    @staticmethod
    def concatenate_v2(arrays, axis=2):
        return torch.cat(arrays, dim=axis)

    @classmethod
    def sqrt(cls, x, dtype=None):
        if isinstance(x, (float, int)):
            x = torch.tensor(x, dtype=dtype)
        elif dtype is not None:
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            x = x.type(dtype)
        return torch.sqrt(x)

    @classmethod
    def mean(cls, x, axis=-1, keepdims=True):
        return x.mean(axis, keepdim=keepdims)

    @classmethod
    def conj(cls, x, inplace=False):
        torch110 = bool(int(torch.__version__.split('.')[1]) >= 10)
        if inplace and (not torch110 and getattr(x, 'requires_grad', False)):
            raise Exception("Torch autograd doesn't support `out=`")
        if inplace:
            out = (torch.conj(x) if torch110 else
                   torch.conj(x, out=x))
        else:
            if torch110:
                x = x.detach().clone()
            out = (torch.conj(x) if cls._is_complex(x) else
                   x)
        return out

    @classmethod
    def zeros_like(cls, ref, shape=None):
        shape = shape if shape is not None else ref.shape
        return torch.zeros(shape, dtype=ref.dtype, layout=ref.layout,
                           device=ref.device)

    @classmethod
    def reshape(cls, x, shape):
        return x.reshape(*shape)

    @classmethod
    def transpose(cls, x, axes):
        return x.permute(*axes)

    @classmethod
    def assign_slice(cls, x, x_slc, slc):
        x[slc] = x_slc
        return x
