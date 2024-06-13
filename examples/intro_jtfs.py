# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
Joint Time-Frequency Scattering Introduction
============================================
  1. Transform a trumpet signal
  2. Visualize coefficients and filterbank
  3. Normalize coefficients
  4. Feed to simple PyTorch 1D CNN
"""

###############################################################################
# Import the necessary packages
# -----------------------------

import numpy as np
import torch
import torch.nn as nn
from wavespin import TimeFrequencyScattering1D
from wavespin.visuals import viz_jtfs_2d
from wavespin.toolkit import normalize

#%%############################################################################
# Load trumpet and create scattering object
# -----------------------------------------
# Load trumpet, duration 2.9 seconds (sampling rate, fs=22050)
# generated via `librosa.load(librosa.ex('trumpet'))[0][:int(2.9*22050)]`
x = np.load('librosa_trumpet.npy')
N = x.shape[-1]

# 10 temporal octaves
J = 10
# 16 bandpass wavelets per octave
# J*Q ~= 160 total temporal coefficients in first-order scattering
Q = 16
# scale of temporal invariance, .93 ms (2**11 [samples] / fs [samples/sec])
T = 2**11
# 4 frequential octaves
J_fr = 4
# 1 bandpass wavelet per octave
Q_fr = 1
# scale of frequential invariance, F/Q == 0.5 cycle per octave
F = 8
# average to reduce transform size and impose freq transposition invariance
average_fr = True
# return packed as dict keyed by pair names for easy inspection
out_type = 'dict:array'

configs = dict(J=J, shape=N, Q=Q, T=T, J_fr=J_fr, Q_fr=Q_fr, F=F,
               average_fr=average_fr, out_type=out_type)
jtfs = TimeFrequencyScattering1D(**configs, frontend='numpy')

#%%############################################################################
# Scatter
# -------
Scx = jtfs(x)

# print pairs and shapes; see `jtfs_2d_cnn` example regarding `n2` etc
print("(batch_size, n2*n1_fr, n1, t):")
for pair, c in Scx.items():
    print(tuple(c.shape), '--', pair)

#%%############################################################################
# Visualize
# ---------

# `True` better represents all geometries by setting the maximum within each
# pair to `1`, as lowpassed pairs tend to dominate coefficient norm.
# Also try `False`
equalize_pairs = True

viz_jtfs_2d(jtfs, Scx, viz_coeffs=True, viz_filterbank=True, fs=22050,
            equalize_pairs=equalize_pairs)

###############################################################################
# We observe a fair amount of assymetry in energy concentrations across
# spinned pairs, on slice-per-slice basis, consistent with the slopes observed
# in first-order scattering in the "Intro Scattering" example. JTFS can
# work its magic here.

#%%############################################################################
# Print relevant info
# -------------------
jtfs.info()

#%%############################################################################
# Feed to simple 1D conv-net
# --------------------------
# Minimal network
class Net(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, out_channels=32, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)  # global avg
        self.fc   = nn.Linear(32, 2)  # e.g. binary classification

    def forward(self, x):
        x = self.pool(self.conv(x)).squeeze(-1)  # drop time dim
        return self.fc(x)

# reinitialize in torch backend
configs['out_type'] = 'array'  # pack everything into one tensor
jtfst = TimeFrequencyScattering1D(**configs, frontend='torch')
xt = torch.from_numpy(x)
Scx = jtfst(xt)

# drop zeroth-order, generally uninformative for audio
Scx = Scx[:, 1:]
# channel-norm (mu=None) for 1D convs (axes=-1), log norm (log1p)
Scx = normalize(Scx, mu=None, std_axis=-1, mean_axis=-1)
# print stats
print("{:.1f}, {:.1f} -- mean, std".format(Scx.mean(), Scx.std()))

# initialize network
n_paths = Scx.shape[1]
net = Net(n_paths)

# get outputs, backprop
out = net(Scx)
loss = out.mean()
loss.backward()

# confirm gradients
g = net.conv.weight.grad
print(g.shape, "-- Conv1D weights grad shape")
print(torch.abs(g).mean(), "-- Conv1D weights grad absolute mean")

#%%############################################################################
# Further reading
# ---------------
# This example is kept brief for minimality, and is meant to be read alongside
# the main JTFS overview. That, and others:
#
#   - Main JTFS overview: https://dsp.stackexchange.com/a/78623/50076
#   - `Applied Design <applied_design.html>`_ is the most natural
#     continuation.
#   - "Chirp-sine-pulse" example in README, "https://github.com/gptanon/wttest
#     JTFS has several "pairs", explained there.
#   - `JTFS 2D Conv-Net <jtfs_2d_cnn.html>`_ example, also covering "pairs",
#     printing shapes, and 3D structure.

#%%############################################################################
# I don't know what I'm doing!
# ----------------------------
# That's tough in JTFS, but the following should cover many use cases -
# `jtfs = TimeFrequencyScattering1D(shape=len(x), **cfg)`, with
#
#   1. `cfg=dict(Q=16)` for balanced time-frequency resolution, and detailed
#      joint coefficients (no frequential averaging). All other parameters will
#      be nicely defaulted.
#   2. `1` with `average_fr=True` to reduce output size and impose
#      frequency-shift invariance, but lose some information.
#   3. `2` with `F='global'` to collapse the frequency dimension and drastically
#      reduce output size, with some compute speedup.
#   4. `cfg=dict(Q=16, out_3D=True, average_fr=True, F=8)` for an
#      information-rich, slice-grouped output of reasonable size.
#   5. `cfg=dict(Q=16, average_fr=True, F='global', T='global')` for the smallest
#      possible transform size for a given time-frequency spec.
#
# `1`, `2`, and `3` are for when we just want a big array of numbers to do
# stuff with. `4` exposes an additional degree of freedom and enforces a
# structure, can extract more advanced features. `5` is for pre-computing
# and feeding to a Raspberry Pi, or some other desperation.
#
#   - `Q=8` can replace `Q=16` above for high time resolution.
#   - `Q=24` for high frequency resolution.
#   - `T='global'` is also a decent option for reducing output size at
#     expense of information; in some ML cases, it's even an improvement.
