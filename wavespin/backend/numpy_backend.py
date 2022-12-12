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
    This is a modification of `kymatio/backend/numpy_backend.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    _np = numpy
    _fft = scipy.fft
    _fft_kwargs = {'workers': -1}  # multiprocessing for scipy

    name = 'numpy'

    @staticmethod
    def input_checks(x):
        if x is None:
            raise TypeError('The input should be not empty.')

    @classmethod
    def complex_check(cls, x):
        if not cls._is_complex(x):
            raise TypeError('The input should be complex, got %s' % x.dtype)

    @classmethod
    def real_check(cls, x):
        if not cls._is_real(x):
            raise TypeError('The input should be real and float (got %s).' %
                            x.dtype)

    @classmethod
    def _is_complex(cls, x):
        return x.dtype in (cls._np.complex64, cls._np.complex128)

    @classmethod
    def _is_real(cls, x):
        return x.dtype in (cls._np.float32, cls._np.float64)

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
    def multiply(cls, A, B):
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
    def assign_slice(cls, x, x_slc, slc, axis=None):
        if axis is not None:
            raise NotImplementedError
        x[slc] = x_slc
        return x

    @classmethod
    def cast(cls, x, dtype):
        return x.astype(dtype)

    @classmethod
    def ensure_dtype(cls, x, dtype):
        if not str(x.dtype).endswith(dtype):
            x = cls.cast(x, dtype)
        return x

    @classmethod
    def try_squeeze(cls, x, axis=None):
        """Will squeeze dimensions if they're singular, else return as-is."""
        if axis is None:
            return x.squeeze()
        elif x.shape[axis] == 1:
            return x.squeeze(axis=axis)
        return x
