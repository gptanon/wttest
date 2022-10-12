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
Examples to study to improve understanding of JTFS.

Script is meant to be ran interactively, with IPython and iteractive debugger.

    - `IS_JTFS=True` for JTFS, `False` for time scattering.
    - `IS_AMCOS=True` for amplitude-modulated cosine, `False` for
       exponential chirp.
    - `SHOW_ANSWERS=True` will print answers from a text file, and show relevant
       visuals.
"""

###############################################################################
# Case: AM cosine
# ---------------
#
#    1. fc: how's changing it affect S1D & JTFS coeffs?
#           Do peaks shift across n2 or n1_fr for JTFS?
#    2. fa: how's changing it affect S1D & JTFS coeffs?
#           Do peaks shift across n2 or n1_fr for JTFS?
#    3. Predict S1D for fa=0 (pure sine). What will S0, S1, and S2 look like?
#    4. Predict JTFS for fa=0. What will spinned coeffs look like?
#    5. Why do answers to 3, 4 change with `analytic=False`?

###############################################################################
# Case: Echirp:
# -------------
#
#    1. How's "discrimination" work? Why does one spin resonate strongly
#       but not other?
#    2. Let slc_max = highest energy spinned coef.
#       Let t_max = index of highest energy slice of slc_max.
#       What does unaveraged, complex `slc_max(:, t_max)` look like? i.e.
#       temporal slice along frequency of the most activated spinned `(n2, n1_fr)`
#       slice.
#    3. Why's there still activations in the weaker spin? What is its biggest
#       contributor?
#    4. Why is `pad_mode="reflect"` (along time) a bad idea for JTFS (of any x)?
#    5. (Advanced, guess is OK) Is `E_dn / E_up = inf` possible?

###############################################################################
# Tips
# ----
#
#    1. Insert `imshow`s (`wavespin.visuals.imshow`) into
#       `wavespin/scattering1d/core/timefrequency_scattering1d.py` in
#       places of interest. For example,
#
#       ::
#           imshow(U_2_m.squeeze(), abs=1,
#                  title="n2=%s, n1_fr=%s, spin=%s" % (n2, n1_fr, spin))
#
#       after `U_2_m = B.modulus(Y_fr_c)` in `_frequency_scattering()`, to
#       visualize individual unaveraged joint slices, and do the same right after
#       `Y_2_arr = _right_pad()` to visualize the input to frequential scattering.
#
#    2. `plot(x, complex=2)` shows real, imag, & envelope at once.
#
#    3. We can skip to a specific slice by breakpointing at `1==1` after inserting
#
#       ::
#           if n2 == 5 and n1_fr == 7 and spin == -1:
#               1 == 1
#
#       Check total number of filters with
#
#       ::
#           print(len(jtfs.psi1_f))
#           print(len(jtfs.psi2_f))
#           print(len(jtfs.psi1_f_fr_up[0]))
#
#    4. If a line causes a bug, we can skip to it as:
#
#       ::
#           try:
#               error()
#           except:
#               # whatever, maybe `error()` again to step into it
#
#       Can also be used to get to a line faster (since debugger slows down
#       the runtime) using `1/0`.
#
#    5. The basic debug workflow is to trace a troublesome coefficient from output
#       back to input, plotting coefficients and filters at every computation
#       step, and printing relevant parameters or attributes. If a filter is
#       the problem, we breakpoint into the relevant build stage.
#
#    6. In Spyder, pack all import statements into one for convenience in debug
#       mode, e.g.
#
#       ::
#           from wavespin.visuals import plot, plotscat, imshow
#           from wavespin.scattering1d.filter_bank import morlet_1d
#
#       then type `from w` (or whatever fewest letters work), press arrow up key,
#       and the statement should show up. Other IDEs may have a similar feature.

###############################################################################
# Import the necessary packages
# -----------------------------

import numpy as np
from wavespin import Scattering1D, TimeFrequencyScattering1D
from wavespin.toolkit import echirp
from wavespin.visuals import viz_jtfs_2d
from copy import deepcopy

IS_JTFS = 1
IS_AMCOS = 1
SHOW_ANSWERS = 1

#%%###########################################################################
# Define generators
# -----------------
def gen_amcos(version=0):
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

def gen_echirp():
    # fmin not too low else we get discretization artifacts
    return echirp(N, fmin=20)

def gen_jtfs(**cfg):
    return TimeFrequencyScattering1D(**cfg, average_fr=1, out_type="dict:array",
                                     out_3D=1)

def gen_ts(**cfg):
    return Scattering1D(**cfg)

#%%############################################################################
# Create signal and scattering network
# ------------------------------------
# Refer to the "Intro to JTFS" example for parameter descriptions.
N = 2049
cfg = dict(
    shape=N,
    J=8,
    Q=8,
    T=256,
    pad_mode="reflect",  # required for perfect AM cos
    max_pad_factor=0,    # pads to current power of 2, i.e. to 4096
)

sc = gen_jtfs(**cfg) if IS_JTFS  else gen_ts(**cfg)
x  = gen_amcos()     if IS_AMCOS else gen_echirp()

#%%############################################################################
# Scatter
# -------
out = sc(x)

#%%############################################################################
# Print answers
# -------------
if SHOW_ANSWERS:
    with open('jtfs_case_studies_answers.txt', 'r') as f:
        print(f.read())

#%%############################################################################
# Show accompanying visuals: only JTFS with AM cos
# ------------------------------------------------

if SHOW_ANSWERS:
    # effectively disable averaging, since our studies concern localization.
    # subsampling is hence also disabled, so below is slow.
    cfg['T'] = 1
    cfg['F'] = 1
    jtfs = gen_jtfs(**cfg)
    x0 = gen_amcos(0)
    x1 = gen_amcos(1)
    x2 = gen_amcos(2)
    x3 = gen_amcos(3)

    o0 = jtfs(x0)
    o1 = jtfs(x1)
    o2 = jtfs(x2)
    o3 = jtfs(x3)

#%%
if SHOW_ANSWERS:
    def zero_unspinned(o):
        o = deepcopy(o)
        for pair in o:
            if not pair.endswith('up') and not pair.endswith('dn'):
                o[pair] *= 0
        return o

    o0s = zero_unspinned(o0)
    o1s = zero_unspinned(o1)
    o2s = zero_unspinned(o2)
    o3s = zero_unspinned(o3)

    ckw = dict(viz_filterbank=0, axis_labels=0)
#%%
# All
if SHOW_ANSWERS:
    viz_jtfs_2d(jtfs, o0, **ckw)
#%%
# Spinned only
if SHOW_ANSWERS:
    viz_jtfs_2d(jtfs, o0s, **ckw)
#%%
# Spinned only -- unspinned is similar to before
if SHOW_ANSWERS:
    viz_jtfs_2d(jtfs, o1s, **ckw)
#%%
# Spinned only
# Plot o0s again so we can compare directly against it in Plots pane (Spyder) --
# the idea is for images to be in exact same position on screen, and us have
# ability to switch with a hotkey / click
# Spinned only
if SHOW_ANSWERS:
    viz_jtfs_2d(jtfs, o0s, **ckw)
    viz_jtfs_2d(jtfs, o2s, **ckw)
#%%
# All - unspinned is now different
if SHOW_ANSWERS:
    viz_jtfs_2d(jtfs, o3, **ckw)
#%%
# Spinned only
if SHOW_ANSWERS:
    viz_jtfs_2d(jtfs, o3s, **ckw)
    # explanation:
    for pair in o3s:
        print("%s -- %s" % (o3s[pair].mean(), pair))
