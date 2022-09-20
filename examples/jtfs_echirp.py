# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
Joint Time-Frequency Scattering of a synthetic signal
=====================================================

Compute and visualize JTFS of an exponential chirp and illustrate
FDTS-(frequency-dependent time shifts) discriminability.
"""

###############################################################################
# Import the necessary packages
# -----------------------------
from wavespin.numpy import TimeFrequencyScattering1D
from wavespin.toolkit import echirp
from wavespin.visuals import plot, imshow
from wavespin import visuals

###############################################################################
# Generate echirp and create scattering object
# --------------------------------------------
N = 4096
# span low to Nyquist; assume duration of 1 second
x = echirp(N, fmin=64, fmax=N/2)

# 9 temporal octaves
J = 9
# 8 bandpass wavelets per octave
# J*Q ~= 144 total temporal coefficients in first-order scattering
Q = 16
# scale of temporal invariance, 31.25 ms
T = 2**7
# 4 frequential octaves
J_fr = 4
# 2 bandpass wavelets per octave
Q_fr = 2
# scale of frequential invariance, F/Q == 0.5 cycle per octave
F = 8
# do frequential averaging to enable 4D concatenation
average_fr = True
# frequential padding; 'zero' avoids a few discretization artefacts
# for this example
pad_mode_fr = 'zero'
# return packed as dict keyed by pair names for easy inspection
out_type = 'dict:array'

params = dict(J=J, Q=Q, T=T, J_fr=J_fr, Q_fr=Q_fr, F=F, out_type=out_type,
              average_fr=average_fr, pad_mode_fr=pad_mode_fr)
jtfs = TimeFrequencyScattering1D(shape=N, **params)

###############################################################################
# Take JTFS, print pair names and shapes
# --------------------------------------
Scx = jtfs(x)
print("JTFS pairs:")
for pair in Scx:
    print("{:<12} -- {}".format(str(Scx[pair].shape), pair))

###############################################################################
# Show `x` and its (time-averaged) scalogram
# ------------------------------------------
plot(x, show=1,
     xlabel="time [samples]",
     title="Exponential chirp | fmin=64, fmax=2048, 4096 samples")
imshow(Scx['S1'].squeeze(), abs=1,
       xlabel="time [samples] (subsampled)",
       ylabel="frequency [Hz]",
       title="Scalogram, time-averaged (first-order scattering)")

###############################################################################
# Create & save GIF
# -----------------

# fetch meta (structural info)
jmeta = jtfs.meta()
# specify save folder
savedir = ''
# time between GIF frames (ms)
duration = 200
visuals.gif_jtfs_2d(Scx, jmeta, savedir=savedir, save_images=0, verbose=1,
                    overwrite=1, gif_kw={'duration': duration})

###############################################################################
# Notice how -1 spin coefficients contain nearly all the energy
# and +1 barely any; this is FDTS discriminability.
# For ideal FDTS and JTFS, +1 will be all zeros.
# For down chirp, the case is reversed and +1 hoards the energy.

###############################################################################
# Show joint wavelets on a nicer filterbank
# -----------------------------------------
jtfs_min = TimeFrequencyScattering1D(shape=512, J=5, Q=16, J_fr=3, Q_fr=1,
                                     F=2**3, out_type='dict:array',
                                     # make filterbank smaller for display
                                     paths_exclude={'j2': 1, 'n1_fr': [0, 1]})
visuals.viz_jtfs_2d(jtfs_min, plot_cfg={'filter_part': 'real'})
