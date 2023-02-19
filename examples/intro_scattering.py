# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
Wavelet Scattering Introduction
===============================
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
from wavespin import Scattering1D
from wavespin.visuals import plot, imshow, filterbank_scattering
from wavespin.toolkit import normalize

#%%############################################################################
# Generate trumpet and create scattering object
# ---------------------------------------------
# Load trumpet, duration 2.9 seconds (sampling rate, sr=22050)
# generated via `librosa.load(librosa.ex('trumpet'))[0][:int(2.9*22050)]`
x = np.load('librosa_trumpet.npy')
N = x.shape[-1]

# 10 temporal octaves
J = 10
# 16 bandpass wavelets per octave
# J*Q ~= 160 total temporal coefficients in first-order scattering
Q = 16
# scale of temporal invariance, .93 ms (2**11 [samples] / sr [samples/sec])
T = 2**11

configs = dict(J=J, shape=N, Q=Q, T=T)
sc = Scattering1D(**configs)

#%%############################################################################
# Scatter
# -------
Scx = sc(x)

#%%############################################################################
# Visualize output
# ----------------
meta = sc.meta()
order0_idxs = np.where(meta['order'] == 0)
order1_idxs = np.where(meta['order'] == 1)
order2_idxs = np.where(meta['order'] == 2)

# only 1 coeff for zeroth-order
xlabel = "time index"
plot(Scx[order0_idxs], show=1, xlabel=xlabel, ylabel="amplitude",
     title="Time scattering | Zeroth order", newfig=1)
# show modulus and disable interpolation in few-sample regime (along time axis)
ikw = dict(abs=1, interpolation='none', newfig=1)
imshow(Scx[order1_idxs], **ikw, xlabel=xlabel, ylabel="frequency index",
       title="Time scattering | First order")
imshow(Scx[order2_idxs], **ikw, xlabel=xlabel, ylabel="frequency index",
       title="Time scattering | Second order, unrolled (n2, n1)")

#%%############################################################################
# Visualize filterbank
# --------------------
# Convolution kernels that were used, in frequency domain
filterbank_scattering(sc, second_order=True)

#%%############################################################################
# Print relevant info
# -------------------
sc.info()

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
sct = Scattering1D(**configs, frontend='torch')
xt = torch.from_numpy(x)
Scx = sct(xt)[None]  # add batch dim

# drop zeroth-order, generally uninformative for audio
Scx = Scx[:, 1:]
# channel-norm (mu=None) for 1D convs (axes=-1), log norm (log1p)
Scx = normalize(Scx, mu=None, std_axis=-1, mean_axis=-1)
# print stats
print("{:.1f}, {:.1f} -- (mean, std) of normalized scat coeffs".format(
    Scx.mean(), Scx.std()))

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
# the main scattering overview. That, and others:
#
#   - Main JTFS overview: https://dsp.stackexchange.com/a/78513/50076
#   - `Applied Design <applied_design.html>`_ is the most natural
#     continuation.
#   - `Parameter Sweeps <parameter_sweeps.html>`_

#%%############################################################################
# I don't know what I'm doing!
# ----------------------------
# Throw these at it, something should stick!
#
# With `sc = Scattering1D(shape=len(x), **cfg)`:
#
#   1. `cfg=dict(Q=16)` for balanced time-frequency resolution. All other
#      parameters will be nicely defaulted.
#   2. `1` with `T='global'` to drastically reduce output size.
#   3. `cfg=dict(T=log2(N) - 6)` to have 64 (`= 2**6`) output time units,
#      good if paired with `Q <= 12`.
#
# `1` and `2` are for when we just want a big array of numbers to do stuff with.
# `3` is the same except we expect subsequent processing steps to do better than
# scattering at handling extra time units, which is doable with conv-nets and
# enough data.
#
#   - `Q=8` can replace `Q=16` above for high time resolution.
#   - `Q=24` for high frequency resolution.

#%%############################################################################
# Fundamentals
# ------------
# Everything in this library, except this subsection, assumes familiarity with
# signal processing basics. Either way, the following may help:
#
#   - `But what is the Fourier Transform? <https://youtu.be/spUNpyF58BY>`_
#     Excellent visual intuition.
#   - `DSP Guide <http://www.dspguide.com/pdfbook.htm>`_, from "Convolution" to
#     "Fourier Transform Pairs". Convolution is awfully explained almost
#     universally - not here. The Discrete Fourier Transform, at the root of all
#     practicality, is also well-explained. Note, the Continuous Wavelet Transform
#     is just convolution (rather, cross-correlation, but they're closely related)
#     with wavelets, which is easily understood as multiplication in frequency
#     domain - and scattering builds on CWT.
#   - `The Wavelet Tutorial <https://ccrma.stanford.edu/~unjung/mylec/WTpart1.html>`_,
#     Great intuition for CWT and time-frequency. Though I don't agree with
#     *everything* there.
#   - Bonus reading on DFT, my articles:
#     `Effect of zero-padding on spectrum <https://dsp.stackexchange.com/a/70498/50076>`_,
#     `DFT coefficients meaning <https://dsp.stackexchange.com/a/70395/50076>`_
#
