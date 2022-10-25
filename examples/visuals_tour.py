# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
Visuals tour
============

Show every major use case of most `wavespin.visuals`.
"""

###############################################################################
# Select visuals
# --------------
SHOW = [
    'heatmap',
    'filterbank_scattering',
    'filterbank_jtfs_1d',
    'scalogram',
    'gif_jtfs_2d',
    'gif_jtfs_3d',
    'energy_profile_jtfs',
    'coeff_distance_jtfs',
    'viz_jtfs_2d',
    'viz_spin_1d',
    'viz_spin_2d',
]

#%%############################################################################
# Import the necessary packages, configure
# ----------------------------------------
import numpy as np
from wavespin import TimeFrequencyScattering1D, Scattering1D
from wavespin import visuals as v
from wavespin import toolkit
from wavespin.utils._examples_utils import show_visual

# `False` to run all visuals. Defaults to showing pre-rendered if the repository
# is cloned, which is needed for documentation builds since compute takes long.
# Both write to present working directory.
TRY_SHOW_PRERENDERED = True

#%%############################################################################
# Generate echirp and create scattering object
# --------------------------------------------
N = 4096
# span low to Nyquist; assume duration of 1 second
x = toolkit.echirp(N, fmin=64, fmax=N/2) / 2
x += np.cos(2*np.pi * 360 * np.linspace(0, 1, N, 1)) / 2
x[N//2-16:N//2+16] += 5

# 9 temporal octaves
J = 9
# 16 bandpass wavelets per octave
# J*Q ~= 144 total temporal coefficients in first-order scattering
Q = 16
# scale of temporal invariance, 125 ms (2**9 [samples] / 4096 [samples/sec])
T = 2**9
# 4 frequential octaves
J_fr = 4
# 2 bandpass wavelets per octave
Q_fr = 1
# scale of frequential invariance, F/Q == 1 cycle per octave
F = 16
# average to reduce transform size and impose freq transposition invariance
average_fr = True
# frequential padding; 'zero' avoids a few discretization artefacts
# for this example
pad_mode_fr = 'zero'
# return packed as dict keyed by pair names for easy inspection
out_type = 'dict:array'
# restrict padding a bit to better zoom on time-domain behavior
max_pad_factor = 0
max_pad_factor_fr = 1

kw_common = dict(shape=N, J=J, Q=Q, T=T, frontend='numpy',
                 max_pad_factor=max_pad_factor)
kw_jtfs = dict(J_fr=J_fr, Q_fr=Q_fr, F=F, pad_mode_fr=pad_mode_fr,
               max_pad_factor_fr=max_pad_factor_fr, **kw_common)
kw_sc = dict(out_type='list', **kw_common)
jtfs = TimeFrequencyScattering1D(**kw_jtfs, average_fr=average_fr,
                                 out_type=out_type)
sc_a = Scattering1D(**kw_sc, average=True)
sc_u = Scattering1D(**kw_sc, average=False)

Scx_j  = jtfs(x)
Scx_sa = sc_a(x)
Scx_su = sc_u(x)

# process arg
SHOW = {k: True for k in SHOW}

#%%############################################################################
# Heatmaps
# --------
if SHOW.get('heatmap', False):
    v.filterbank_heatmap(jtfs, first_order=1, second_order=1, frequential=1,
                         parts='all', w=.9)

#%%############################################################################
# Freq-domain filters, with energies and zoom
# -------------------------------------------
if SHOW.get('filterbank_scattering', False):
    v.filterbank_scattering(jtfs, zoom=0)
    v.filterbank_scattering(jtfs, zoom=5)
    v.filterbank_scattering(jtfs, first_order=0, second_order=1, lp_sum=1)

#%%############################################################################
# JTFS filters, in freq domain
# ----------------------------
if SHOW.get('filterbank_jtfs_1d', False):
    v.filterbank_jtfs_1d(jtfs, zoom=0)
    v.filterbank_jtfs_1d(jtfs, zoom=0, both_spins=0)
    v.filterbank_jtfs_1d(jtfs, zoom=-1)
    v.filterbank_jtfs_1d(jtfs, zoom=-1, center_dc=0)

#%%############################################################################
# Simple scalogram
# ----------------
if SHOW.get('scalogram', False):
    v.scalogram(x, sc_u, show_x=1, fs=N)

#%%############################################################################
# GIF of JTFS slices
# ------------------
if SHOW.get('gif_jtfs_2d', False):
    viz_fn = lambda: v.gif_jtfs_2d(Scx_j, jtfs.meta(), verbose=1, show=0,
                                   overwrite=True)
    prerendered_filename = 'jtfs2d.gif'
    show_visual(viz_fn, prerendered_filename, TRY_SHOW_PRERENDERED)

#%%############################################################################
# GIF of full 4D JTFS structure
# -----------------------------
if SHOW.get('gif_jtfs_3d', False):
    viz_fn = lambda: v.gif_jtfs_3d(Scx_j, jtfs, preset='spinned', savedir='',
                                   overwrite=True)
    show_visual(viz_fn, 'jtfs3d.gif', TRY_SHOW_PRERENDERED)

#%%############################################################################
# Energy distribution across pairs and coefficients within
# --------------------------------------------------------
if SHOW.get('energy_profile_jtfs', False):
    _ = v.energy_profile_jtfs(Scx_j, jtfs.meta(), x=x)
    _ = v.energy_profile_jtfs(Scx_j, jtfs.meta(),
                              pairs=('psi_t * psi_f_up', 'psi_t * psi_f_dn'))

#%%############################################################################
# Coefficient *relative* distance on frequency transposition, pairwise
# --------------------------------------------------------------------
if SHOW.get('coeff_distance_jtfs', False):
    f0 = N // 12
    f1 = f0 / np.sqrt(2)
    n_partials = 5
    seg_len = N//8

    x0 = toolkit.fdts(N, n_partials, f0=f0, seg_len=seg_len)[0]
    x1 = toolkit.fdts(N, n_partials, f0=f1, seg_len=seg_len)[0]

    jtfs_x0_all = jtfs(x0)
    jtfs_x1_all = jtfs(x1)
    jtfs_x0_all = toolkit.jtfs_to_numpy(jtfs_x0_all)
    jtfs_x1_all = toolkit.jtfs_to_numpy(jtfs_x1_all)

    _ = v.coeff_distance_jtfs(jtfs_x0_all, jtfs_x1_all, jtfs.meta(), plots=True)
    # note how S1 relative distance is much greater than that of JTFS slices

#%%############################################################################
# JTFS 2D filterbank and coefficients
# -----------------------------------
if SHOW.get('viz_jtfs_2d', False):
    ###########################################################################
    # Configure
    # ---------
    ju_kw = dict(
        # don't average, else JTFS coefficients don't look interesting
        average_fr = False,
        average = False,
        # must compensate for lack of averaging with lack of subsampling
        oversampling_fr = 99,
        oversampling = 99,
        # list required for `average=False`
        out_type = 'dict:list',
        # omit low energy coefficients for more compact visualization
        paths_exclude = {'n2': [3, 4], 'n1_fr': [0]},
    )
    jtfs_u = TimeFrequencyScattering1D(**kw_jtfs, **ju_kw)

    Scx_ju  = jtfs_u(x)
    ckw = dict(jtfs=jtfs_u)

    #%%########################################################################
    # Visualize
    # ---------
    # show the coefficients
    v.viz_jtfs_2d(**ckw, Scx=Scx_ju, viz_filterbank=0, viz_coeffs=1)

    # show the corresponding filters, real part by default
    v.viz_jtfs_2d(**ckw, viz_filterbank=1,
                  savename='j2d_0')
    # imaginary part
    v.viz_jtfs_2d(**ckw, viz_filterbank=1, plot_cfg={'filter_part': 'imag'},
                  savename='j2d_1')
    # pseudo-complex colormap
    v.viz_jtfs_2d(**ckw, viz_filterbank=1, plot_cfg={'filter_part': 'complex'},
                  savename='j2d_2')

    # show amplitude envelopes only
    v.viz_jtfs_2d(**ckw, viz_filterbank=1,
                  plot_cfg={'filter_part': 'abs',
                            'imshow_kw_filterbank': {'cmap': 'turbo'}},
                  savename='j2d_3')

    # zoom on every wavelet's own support, illustrating self-similarity
    v.viz_jtfs_2d(**ckw, viz_filterbank=1, plot_cfg={'filterbank_zoom': -1})

    # Make gif from images we just made
    v.make_gif(loaddir='', savepath='jtfs2d_filterbank.gif', duration=1500,
               delimiter='j2d', overwrite=1, delete_images=1, HD=1, verbose=1)

#%%############################################################################
# Visualize a single Morlet
# -------------------------
if SHOW.get('viz_spin_1d', False):
    viz_fn = lambda: v.viz_spin_1d(verbose=1, savepath='viz_morlet_1d')
    show_visual(viz_fn, 'viz_morlet_1d.gif', TRY_SHOW_PRERENDERED)

#%%############################################################################
# Visualize JTFS wavelets in 4D
# -----------------------------
if SHOW.get('viz_spin_2d', False):
    viz_fn = lambda: v.viz_spin_2d(preset=0, verbose=1, savepath='viz_spin_up')
    show_visual(viz_fn, 'viz_spin_up.gif', TRY_SHOW_PRERENDERED)

#%%
if SHOW.get('viz_spin_2d', False):
    viz_fn = lambda: v.viz_spin_2d(preset=1, verbose=1, savepath='viz_spin_both')
    show_visual(viz_fn, 'viz_spin_both.gif', TRY_SHOW_PRERENDERED)

#%%
if SHOW.get('viz_spin_2d', False):
    viz_fn = lambda: v.viz_spin_2d(preset=2, verbose=1, savepath='viz_spin_all')
    show_visual(viz_fn, 'viz_spin_all.gif', TRY_SHOW_PRERENDERED)
