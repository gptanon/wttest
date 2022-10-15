# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import numpy
import scipy.fft


class NumPyBackend:
    """
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/backend/
    numpy_backend.py

    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    _np = numpy
    _fft = scipy.fft

    name = 'numpy'

    @staticmethod
    def input_checks(x):
        if x is None:
            raise TypeError('The input should be not empty.')

    @classmethod
    def complex_check(cls, x):
        if not cls._is_complex(x):
            raise TypeError('The input should be complex.')

    @classmethod
    def real_check(cls, x):
        if not cls._is_real(x):
            raise TypeError('The input should be real and float (got %s).' %
                            x.dtype)

    @classmethod
    def _is_complex(cls, x):
        return (x.dtype == cls._np.complex64) or (x.dtype == cls._np.complex128)

    @classmethod
    def _is_real(cls, x):
        return (x.dtype == cls._np.float32) or (x.dtype == cls._np.float64)

    @classmethod
    def concatenate(cls, arrays, axis=-2, keep_cat_dim=False):
        """
        Let `arrays = [randn(1, 3, 8), randn(1, 3, 8)]`:
            `keep_cat_dim=True`  -> (1, 2, 3, 8)
            `keep_cat_dim=False` -> (1, 6, 8)
        """
        fn = (cls._np.stack if keep_cat_dim else
              cls._np.concatenate)
        return fn(arrays, axis=axis)

    @classmethod
    def modulus(cls, x):
        """Complex modulus, i.e. absolute value.

        This method exists in case of future optimizations.
        """
        return cls._np.abs(x)

    @classmethod
    def cdgmm(cls, A, B):
        """Complex pointwise multiplication.

        This method exists in case of future optimizations.
        """
        return A * B

    @classmethod
    def sqrt(cls, x, dtype=None):
        return cls._np.sqrt(x, dtype=dtype)

    @classmethod
    def mean(cls, x, axis=-1, keepdims=True):
        return x.mean(axis, keepdims=keepdims)

    @classmethod
    def conj(cls, x, inplace=False):
        if inplace and cls._is_complex(x):
            out = cls._np.conj(x, out=x)
        elif not inplace:
            out = (cls._np.conj(x) if cls._is_complex(x) else
                   x)
        else:
            out = x
        return out

    @classmethod
    def zeros_like(cls, ref, shape=None):
        shape = shape if shape is not None else ref.shape
        return cls._np.zeros(shape, dtype=ref.dtype)

    @classmethod
    def reshape(cls, x, shape):
        return x.reshape(*shape)

    @classmethod
    def transpose(cls, x, axes):
        return x.transpose(*axes)

    @classmethod
    def assign_slice(cls, x, x_slc, slc):
        x[slc] = x_slc
        return x
