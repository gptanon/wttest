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

    This is a modification of `kymatio/scattering1d/backend/torch_backend.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    @classmethod
    def subsample_fourier(cls, x, sub, axis=-1):
        """See `help(wavespin.scattering1d.backend.numpy_backend)`."""
        # handle common cases for speed
        if sub == 1:
            return x
        elif axis == -1:
            return x.reshape(*x.shape[:-1], sub, -1).mean(dim=x.ndim - 1)

        axis = axis if axis >= 0 else x.ndim + axis  # ensure non-negative
        s = list(x.shape)
        N = s[axis]
        s[axis] = N // sub
        s.insert(axis, sub)

        res = x.reshape(s).mean(dim=axis)
        return res

    @staticmethod
    def pad(x, pad_left, pad_right, pad_mode='reflect', axis=-1):
        """See `help(wavespin.scattering1d.backend.numpy_backend)`."""
        return agnostic.pad(x, pad_left, pad_right, pad_mode, axis=axis)

    @staticmethod
    def unpad(x, i0, i1, axis=-1):
        """See `help(wavespin.scattering1d.backend.numpy_backend)`."""
        if axis == -1:
            return x[..., i0:i1]
        return x[agnostic.index_axis(i0, i1, axis, x.ndim)]

    @classmethod
    def fft(cls, x, axis=-1):
        return torch.fft.fft(x, dim=axis)

    @classmethod
    def ifft(cls, x, axis=-1):
        return torch.fft.ifft(x, dim=axis)

    @classmethod
    def r_fft(cls, x, axis=-1):
        return torch.fft.fft(x, dim=axis)

    @classmethod
    def ifft_r(cls, x, axis=-1):
        return torch.fft.ifft(x, dim=axis).real

    @classmethod
    def conj_reflections(cls, x, ind_start, ind_end, k, N, J_pad, log2_T, J,
                         pad_left, pad_right, trim_tm):
        return agnostic.conj_reflections(
            x, ind_start, ind_end, k, N, J_pad, log2_T, J,
            pad_left, pad_right, trim_tm)


backend = TorchBackend1D
