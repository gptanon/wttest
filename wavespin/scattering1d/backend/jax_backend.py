# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
from .numpy_backend import NumPyBackend1D
from ...backend.jax_backend import JaxBackend


class JaxBackend1D(NumPyBackend1D, JaxBackend):
    # Simply inherit everything as all used `numpy.` operations have
    # `jax.numpy.` equivalents.
    pass


backend = JaxBackend1D
