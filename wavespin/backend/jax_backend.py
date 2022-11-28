# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
from jax import numpy
from .numpy_backend import NumPyBackend


class JaxBackend(NumPyBackend):
    """Jax backend base class.

    Since all of used NumPy operations have Jax equivalents, having a Jax backend
    amounts to a simple inheritance, except where divergent behavior must
    be handled:

        https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html

    Note that JIT-ing isn't implemented.
    """
    _np = numpy
    _fft = numpy.fft  # need numpy's for compat with `jax.numpy`
    _fft_kwargs = {}  # since `numpy.fft` lacks `workers`

    name = 'jax'

    @classmethod
    def sqrt(cls, x, dtype=None):
        # `jax.numpy`'s `dtype` won't work as of jax 0.3.25
        return cls._np.sqrt(cls._np.asarray(x, dtype=dtype))

    @classmethod
    def assign_slice(cls, x, x_slc, slc, axis=None):
        if axis is not None:
            raise NotImplementedError
        # jax's alternative to in-place assignment
        x = x.at[slc].set(x_slc)
        return x
