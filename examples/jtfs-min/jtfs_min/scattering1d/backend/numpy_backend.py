# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
from ...backend.numpy_backend import NumPyBackend


class NumPyBackend1D(NumPyBackend):
    """NumPy backend object.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/backend/
    numpy_backend.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    @classmethod
    def subsample_fourier(cls, x, k, axis=-1):
        """Subsampling in the Fourier domain.

        Subsampling in the temporal domain amounts to folding in the Fourier
        domain, so the input is periodized according to the subsampling factor.
        https://dsp.stackexchange.com/a/74734/50076

        Parameters
        ----------
        x : tensor
            Input tensor, with length divisible by `k`.
        k : int
            The subsampling factor.
        axis : int
            Axis along which to subsample.

        Returns
        -------
        res : tensor
            The input tensor subsampled along `axis`, yielding a tensor of size
            `x.shape[axis] // k` along that axis.
        """
        if k == 1:
            return x

        axis = axis if axis >= 0 else x.ndim + axis  # ensure non-negative
        s = list(x.shape)
        N = s[axis]
        re = (k, N // k)
        s.pop(axis)
        s.insert(axis, re[1])
        s.insert(axis, re[0])

        res = x.reshape(s).mean(axis=axis)
        return res

    @classmethod
    def pad(cls, x, pad_left, pad_right, pad_mode='reflect', axis=-1):
        """Pad N-dim arrays along any one axis.

        Parameters
        ----------
        x : tensor
            Input array.

        pad_left : int
            Amount to add on the left of the tensor (beginning of the axis).

        pad_right : int
            Amount to add on the right of the tensor (end of the axis).

        pad_mode : str
            Type of padding to use. 'zero' or 'reflect'.

        Returns
        -------
        output : tensor
            The padded array.
        """
        if pad_mode == 'zero':
            pad_mode = 'constant'

        paddings = [(0, 0)] * x.ndim
        paddings[axis] = (pad_left, pad_right)

        output = cls._np.pad(x, paddings, mode=pad_mode)
        return output

    @staticmethod
    def unpad(x, i0, i1, axis=-1):
        """Unpad N-dim tensor along one dimension.

        Slices the input tensor at indices between i0 and i1 along any one axis.

        Parameters
        ----------
        x : tensor
            Input with at least one axis.
        i0 : int
            Start of original signal before padding.
        i1 : int
            End of original signal before padding.
        axis : int
            Axis to unpad.

        Returns
        -------
        x_unpadded : tensor
            The unpadded tensor. If `axis=-1`, `x[..., i0:i1]`.
        """
        if axis == -1:
            return x[..., i0:i1]
        elif axis == -2:
            slc = (slice(None),) * (x.ndim + axis) + (slice(i0, i1),)
            return x[slc]

    @classmethod
    def fft(cls, x, axis=-1):
        return cls._fft.fft(x, axis=axis, workers=-1)

    @classmethod
    def ifft(cls, x, axis=-1):
        return cls._fft.ifft(x, axis=axis, workers=1)

    @classmethod
    def r_fft(cls, x, axis=-1):
        """FFT of real-valued inputs, first casting to complex if necessary."""
        return cls._fft.fft(x, axis=axis, workers=1)

    @classmethod
    def ifft_r(cls, x, axis=-1):
        """iFFT followed by taking real part."""
        return cls._fft.ifft(x, axis=axis, workers=-1).real


backend = NumPyBackend1D
