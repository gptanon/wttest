# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
JTFS Transform Case Studies
===========================
Reproduced for minimal JTFS. See `wavespin/examples/more/jtfs_case_studies.py`.
"""

###############################################################################
# Import the necessary packages
# -----------------------------

import numpy as np
from wavespin.toolkit import echirp
from jtfs_min import TimeFrequencyScattering1D

IS_AMCOS = 0

#%%###########################################################################
# Define generators
# -----------------
def gen_amcos(N, version=0):
    # carrier, modulator
    fc, fa = {
        0: (128, 4),
        1: (256, 4),
        2: (128, 8),
        3: (128, 0),
    }[version]
    # endpoint required for perfect AM cos with `pad_mode='reflect'`
    t = np.linspace(0, 1, N, endpoint=True)
    c = np.cos(2*np.pi * fc * t)
    a = np.cos(2*np.pi * fa * t)
    return a * c

def gen_echirp(N):
    # fmin not too low else we get discretization artifacts
    return echirp(N, fmin=20)

#%%############################################################################
# Create signal and scattering network
# ------------------------------------
# Refer to the "Intro to JTFS" example for parameter descriptions.
N = 2049
jtfs = TimeFrequencyScattering1D(N, J=8, Q=8, T=256)
x = gen_amcos(N) if IS_AMCOS else gen_echirp(N)

#%%############################################################################
# Scatter
# -------
out = jtfs(x)
