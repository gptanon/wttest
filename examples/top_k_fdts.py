# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
FDTS localization and slopes
============================
  - Compute JTFS of FDTS (frequency-dependent time-shifted) waveforms
  - Show that they can be localized under severe noise and mild averaging
  - Visualize the localization and show associated coefficient slopes
"""

###############################################################################
# Import the necessary packages
# -----------------------------
import os
import numpy as np
from scipy.signal import tukey

from wavespin import Scattering1D, TimeFrequencyScattering1D
from wavespin.visuals import viz_top_fdts, scalogram
from wavespin.toolkit import echirp, fdts

# set animation save directory; defaults to current working directory
SAVEDIR = ''

#%%############################################################################
# Build JTFS instance
# -------------------

# take duration of 1 second
fs = 2048
N = fs

common_params = dict(
    shape=N,
    # good tradeoff in time-frequency resolution for considered signals
    Q=8,
    # don't need width greater than 2**8=256 samples for considered signals
    J=(8, 8),
    # prioritize accuracy
    max_pad_factor=None,
    frontend='numpy',
)

jtfs_kw = dict(
    # keep averaging moderate to not lose too much localization
    T = 128,
    F = 4,
    # high spatial localization along freq, and keep number of slices low
    Q_fr = 1,
    # scalogram size is <64 so don't need width more than 2**4=16
    J_fr = 4,
    # need for `pack_coeffs_jtfs`
    out_type = 'dict:array',
    # best results for applications utilizing 4D structure
    out_3D = True,
    # show with frequential averaging (unfavorable case)
    average_fr = True,
    # improved FDTS discriminability in general case
    pad_mode_fr = 'conj-reflect-zero',
    # maximize total number of coefficients just to be safe
    sampling_filters_fr = 'resample',
    # keep results consistent in case of changes to the Smart Paths algorithm
    smart_paths = 'primitive',
    # omit uninteresting coefficients (high freq second order, low energy)
    paths_exclude={'n2': [2, 3, 4, 5]},
)

jtfs = TimeFrequencyScattering1D(**common_params, **jtfs_kw)
# Also build corresponding scalogram generator
sc = Scattering1D(**common_params, max_order=1, out_type='list', average=False,
                  oversampling=99)

#%%############################################################################
# Build the FDTS signal
# ---------------------
t = np.linspace(0, 1, N, 0)

specs = [
  # `tmin, tmax, fmin, fmax` args to `echirp`
  (.75, .9, .05, .9),
  (.1, .2, .3, .1),
  (.25, .9, .015/4, .04/3),
]

# generate Tukey-windowed echirps, i.e. straight lines in time-log-frequency
# plane. Segment length along time is determined by `tmin` and `tmax` (which is
# then zero-padded to full length), and along frequency by `fmin` and `fmax`.
x = np.zeros(N)
for spec in specs:
    tmn0, tmx0, fmn0, fmx0 = spec
    fmn0, fmx0 = fmn0 * N/2, fmx0 * N/2

    idxs = np.where((tmn0 < t) * (t < tmx0))[0]
    N0 = len(idxs)
    pl0 = idxs[0]
    pr0 = N - idxs[-1] - 1
    print(idxs[:5], pl0, pr0)

    y = echirp(N0, fmin=fmn0, fmax=fmx0, tmin=tmn0, tmax=tmx0)
    y *= tukey(len(y))
    y = np.pad(y, [pl0, pr0])
    x += y

# add FDTS partials: windowed pure sines, time-shifted to make a sloped line
# in log-frequency
x += fdts(N, n_partials=4, f0=100, partials_f_sep=1.6, total_shift=-150,
          seg_len=N//8)[1]
# add noise
x_clean = x
np.random.seed(0)
noise = np.random.randn(N) / 1.2
x = x_clean + noise

# SNR is computed as 10*log10(x.var() / noise.var()), where x.var() is taken
# within relevant bandwidth. Each signal isolated in the time-log-frequency plane
# has variance of about 0.5, or a pure sine, so we use that.
snr = 10*np.log10(.5 / noise.var())

#%%############################################################################
# Show clean and noised signals' scalograms
# -----------------------------------------
plot_cfg = {'imshow_kw': dict(interpolation='none')}
skw = dict(w=1.4, h=1.1, fs=fs, plot_cfg=plot_cfg)

plot_cfg['title_scalogram'] = "Scalogram(x_clean)"
scalogram(x_clean, sc, **skw)
plot_cfg['title_scalogram'] = "Scalogram(x), SNR=%.3g dB" % snr
scalogram(x, sc, **skw)

#%%############################################################################
# Make initial sketch by showing many coefficients
# ------------------------------------------------
idxs = None
ckw = dict(jtfs=jtfs, x=x, fs=fs, wav_zoom=2)
top_k = 20  # arbitrary
close_figs = False  # for online documentation

_ = viz_top_fdts(**ckw, idxs=idxs, top_k=top_k, close_figs=close_figs,
                 render='show')

#%%############################################################################
# Handpick top 4 best coefficients, animate
# -----------------------------------------
idxs = [20, 40, 33, 16]
render_kw = {'duration': 2000}
savepath = os.path.join(os.path.abspath(SAVEDIR), 'top_k_fdts.gif')

_ = viz_top_fdts(**ckw, idxs=idxs, render_kw=render_kw, savepath=savepath,
                 render='gif')
# NOTE: higher quality GIF generation may be possible,
# see `wavespin.visuals.make_gif`. ReadTheDocs will generate lower quality.
