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
    def zeros_like(cls, ref, shape=None):
        shape = shape if shape is not None else ref.shape
        return cls._np.zeros(shape, dtype=ref.dtype)

