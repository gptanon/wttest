# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
=== DO NOT EDIT! (unless you know what you're doing) ==========================

Internal parameters for Scattering1D and TimeFrequencyScattering1D.
"""

C_S1D = dict(
    sigma0=.13,
    P_max=5,
    eps=1e-7,
    criterion_amplitude=1e-3,
)
C_JTFS = dict(
    sigma_max_to_min_max_ratio=1.2,
    width_exclude_ratio=0.5,
    N_fr_p2up=None,
    N_frs_min_global=8,
    **C_S1D,
)
