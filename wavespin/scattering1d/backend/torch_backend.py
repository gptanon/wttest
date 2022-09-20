# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import torch
import torch.fft
from ...backend.torch_backend import TorchBackend
from . import agnostic_backend as agnostic


class TorchBackend1D(TorchBackend):
    """PyTorch backend object.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/backend/
    torch_backend.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    @classmethod
    def subsample_fourier(cls, x, k, axis=-1):
        """See `help(wavespin.scattering1d.backend.numpy_backend)`."""
        if k == 1:
            return x

        axis = axis if axis >= 0 else x.ndim + axis  # ensure non-negative
        s = list(x.shape)
        N = s[axis]
        re = (k, N // k)
        s.pop(axis)
        s.insert(axis, re[1])
        s.insert(axis, re[0])

        res = x.reshape(s).mean(dim=axis)
        return res

    @staticmethod
    def pad(x, pad_left, pad_right, pad_mode='reflect', axis=-1):
        """See `help(wavespin.scattering1d.backend.numpy_backend)`."""
        return agnostic.pad(x, pad_left, pad_right, pad_mode, axis=axis)

    @staticmethod
    def unpad(x, i0, i1, axis=-1):
        """See `help(wavespin.scattering1d.backend.numpy_backend)`."""
        return x[agnostic.index_axis(i0, i1, axis, x.ndim)]

    @classmethod
    def fft(cls, x, axis=-1):
        return torch.fft.fft(x, dim=axis)

    @classmethod
    def rfft(cls, x, axis=-1):
        cls.real_check(x)

        return torch.fft.fft(x, dim=axis)

    @classmethod
    def irfft(cls, x, axis=-1):
        cls.complex_check(x)

        return torch.fft.ifft(x, dim=axis).real

    @classmethod
    def ifft(cls, x, axis=-1):
        cls.complex_check(x)

        return torch.fft.ifft(x, dim=axis)

    @classmethod
    def conj_reflections(cls, x, ind_start, ind_end, k, N, pad_left, pad_right,
                         trim_tm):
        return agnostic.conj_reflections(cls, x, ind_start, ind_end, k, N,
                                         pad_left, pad_right, trim_tm)


backend = TorchBackend1D
