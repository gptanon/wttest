# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import tensorflow as tf

from ...backend.tensorflow_backend import TensorFlowBackend
from . import agnostic_backend as agnostic


class TensorFlowBackend1D(TensorFlowBackend):
    """TensorFlow backend object.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/backend/
    tensorflow_backend.py
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

        res = tf.reduce_mean(tf.reshape(x, s), axis=axis)
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
        cls.complex_check(x)
        x = cls._maybe_transpose_for_fft(x, axis)

        out = tf.signal.fft(x, name='fft_1d')
        return cls._maybe_transpose_for_fft(out, axis)

    @classmethod
    def ifft(cls, x, axis=-1):
        cls.complex_check(x)
        x = cls._maybe_transpose_for_fft(x, axis)

        out = tf.signal.ifft(x, name='ifft_1d')
        return cls._maybe_transpose_for_fft(out, axis)

    @classmethod
    def r_fft(cls, x, axis=-1):
        # TF won't auto-cast, check input is real so `cdtype` won't fail
        cls.real_check(x)
        x = cls._maybe_transpose_for_fft(x, axis)

        cdtype = {'float32': 'complex64', 'float64': 'complex128'
                  }[x.dtype.name]
        x = cls.cast(x, cdtype)

        out = tf.signal.fft(x, name='r_fft_1d')
        return cls._maybe_transpose_for_fft(out, axis)

    @classmethod
    def ifft_r(cls, x, axis=-1):
        cls.complex_check(x)
        x = cls._maybe_transpose_for_fft(x, axis)

        out = tf.math.real(tf.signal.ifft(x, name='ifft_r_1d'))
        return cls._maybe_transpose_for_fft(out, axis)

    @classmethod
    def conj_reflections(cls, x, ind_start, ind_end, k, N, pad_left, pad_right,
                         trim_tm):
        return agnostic.conj_reflections(x, ind_start, ind_end, k, N,
                                         pad_left, pad_right, trim_tm)

    @classmethod
    def _maybe_transpose_for_fft(cls, x, axis):
        if axis in (-2, x.ndim - 2) and x.ndim > 2:
            D = x.ndim
            x = tf.transpose(x, (*list(range(D - 2)), D - 1, D - 2))
        elif axis not in (-1, x.ndim - 1):
            # -1 means no need to transpose
            raise NotImplementedError("`axis` must be -1 or -2")
        return x


backend = TensorFlowBackend1D
