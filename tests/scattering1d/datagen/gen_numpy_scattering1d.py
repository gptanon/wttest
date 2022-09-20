# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Generate data for the numpy Scattering1D output test."""
import numpy as np
from wavespin import Scattering1D

# generate data, set params
np.random.seed(0)
x = np.random.randn(2048)

J = 9
Q = 8
N = x.shape[-1]

# scatter
sc = Scattering1D(N, J, Q, frontend='numpy', smart_paths='primitive')
Sx0 = sc(x)

# store generating source code
with open(__file__, 'r') as f:
    code = f.read()

# save
np.savez("../data/test_scattering_1d.npz", x=x, J=J, Q=Q, N=N, Sx=Sx0, code=code)
