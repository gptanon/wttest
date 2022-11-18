# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
Parameter Sweeps
================

Show and explain effects of basic parameters on filterbank.
"""
##############################################################################
# Import the necessary packages, configure
# ----------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft, ifftshift

from wavespin import Scattering1D
from wavespin.visuals import plot, imshow, make_gif
from wavespin.utils._examples_utils import show_visual

# `False` to run all visuals. Defaults to showing pre-rendered if the repository
# is cloned, which is needed for documentation builds since compute takes long.
# Both write to present working directory.
TRY_SHOW_PRERENDERED = True
# Unpadded input length for all filterbanks. `TRY_SHOW_PRERENDERED` needs `4096`.
N = 4096
# set save directory
SAVEDIR = ''

#%%###########################################################################
# Helper methods / params
# -----------------------

def viz_fb(sc1, sc2, title, savename):
    """Reusable visualizer."""
    def group_filts_and_to_time(sc):
        psi_fs = np.array([p[0] for p in sc.psi1_f])
        psi_ts = ifftshift(ifft(psi_fs, axis=-1), axes=-1)
        psi_ts /= np.abs(psi_ts).max(axis=-1)[:, None]
        n_cqt = sum([p['is_cqt'] for p in sc.psi1_f])
        return psi_fs, psi_ts, n_cqt

    psi_fs1, psi_ts1, n_cqt1 = group_filts_and_to_time(sc1)
    psi_fs2, psi_ts2, n_cqt2 = group_filts_and_to_time(sc2)
    N_filt = psi_fs1.shape[-1]

    _title = lambda txt: (txt, {'fontsize': 16})

    # plot ###################################################################
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    imshow(psi_ts1, abs=1, ax=axes[0, 0], fig=fig, show=0,
           title=_title("Filterbank modulus, amplitude-equalized"),
           xlabel="frequencies [samples] | dc, +, -",
           interpolation='none')
    plot(psi_fs1.T[:N_filt//2 + 1], ax=axes[0, 1], fig=fig,
         color='tab:blue',
         title=_title("Filterbank in freq domain, positive freqs"))
    imshow(psi_ts2, abs=1, ax=axes[1, 0], fig=fig, show=0,
           interpolation='none')
    plot(psi_fs2.T[:N_filt//2 + 1], ax=axes[1, 1], fig=fig, color='tab:blue')

    # do suplabel instead of ylabel to avoid spacing shifts for different ranges
    # of yticks
    fig.supylabel("wavelet index", fontsize=18, weight='bold', y=.7, x=-.06)

    # show cqt bound
    _hlines = lambda l: (l, {'linewidth': 2, 'color': 'white'})
    plot([], hlines=_hlines(n_cqt1 - .5), ax=axes[0, 0], fig=fig)
    plot([], hlines=_hlines(n_cqt2 - .5), ax=axes[1, 0], fig=fig)

    # style & save ###########################################################
    fig.suptitle(title, weight='bold', fontsize=20, y=.95)
    fig.subplots_adjust(left=0, right=1, wspace=.07)

    path = os.path.join(SAVEDIR, f"{savename}.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

common_kw = dict(shape=N, smart_paths=0, max_order=1, normalize='l1',
                 max_pad_factor=1)
gif_common_kw = dict(loaddir=SAVEDIR, start_end_pause=2, delete_images=1,
                     verbose=1, HD=True)

_savename = lambda base_name, i: base_name + str(i).zfill(2)
# hack to force failure if we can't load image; the actual `viz_fn` is what's
# below `except ZeroDivisionError`, but we don't pre-wrap for cleaner code.
viz_fn = lambda: 1/0

#%%###########################################################################
# Sweep `J` for two `Q`
# ---------------------
base_name = "J_sweep"

try:
    show_visual(viz_fn, base_name + '.gif', TRY_SHOW_PRERENDERED)
except ZeroDivisionError:
    Q1, Q2 = 4, 16
    for i, J in enumerate(range(2, int(np.log2(N)) + 1)):
        sc1 = Scattering1D(**common_kw, J=J, Q=Q1)
        sc2 = Scattering1D(**common_kw, J=J, Q=Q2)

        title = "J={} | Q={} & {}, r_psi={:.3g}".format(J, Q1, Q2, sc1.r_psi[0])
        viz_fb(sc1, sc2, title=title, savename=_savename(base_name, i))

    make_gif(**gif_common_kw, savepath=base_name + ".gif", duration=750,
              delimiter=base_name)

##############################################################################
# `J` controls the width of the largest wavelet.
#
#   - The white horizontal line is the boundary between CQT and non-CQT wavelets.
#   - Lesser `J` is a *subset* of a larger `J`, except for non-CQT filters.
#     That is, increasing `J` is akin to adding larger temporal width CQT
#     filters, then finishing off, as usual, with non-CQT filters of the (new)
#     largest width.
#   - Larger `J` corresponds to a greater portion of wavelets being CQT,
#     for any `Q`. For the greater `Q`, this portion is lesser - why?
#

#%%###########################################################################
# Sweep `Q` for two `J`
# ---------------------
base_name = "Q_sweep"

try:
    show_visual(viz_fn, base_name + '.gif', TRY_SHOW_PRERENDERED)
except ZeroDivisionError:
    J1, J2 = 6, 10
    for i, Q in enumerate(range(1, 25)):
        sc1 = Scattering1D(**common_kw, Q=Q, J=J1)
        sc2 = Scattering1D(**common_kw, Q=Q, J=J2)

        title = "Q={} | J={} & {}, r_psi={:.3g}".format(Q, J1, J2, sc1.r_psi[0])
        viz_fb(sc1, sc2, title=title, savename=_savename(base_name, i))

    make_gif(**gif_common_kw, savepath=base_name + ".gif", duration=500,
             delimiter=base_name)

##############################################################################
# `Q` controls the number of wavelets per octave, and their frequency resolution.
#
#   - Greater `Q` is greater frequency resolution, i.e. greater temporal width.
#     Hence, the largest scale, `J` is attained sooner (in fewer number of
#     wavelets). It also means that the *smallest* width wavelet is wider,
#     hence the largest and smallest wavelets are closer in width and bandwidth.
#   - For a given `J`, increasing `Q` more and more will stray further and further
#     from the literature's ideal of `J * Q`, as the CQT portion shrinks further.
#     So we don't have "`J` octaves", though we still tile all frequencies.
#

#%%###########################################################################
# Sweep `r_psi` for two `Q`
# -------------------------
base_name = 'r_psi_sweep'

try:
    show_visual(viz_fn, base_name + '.gif', TRY_SHOW_PRERENDERED)
except ZeroDivisionError:
    Q1, Q2 = 4, 16
    n_frames = 24
    J = 10

    for i, r_psi in enumerate(np.linspace(.01, .99, n_frames, endpoint=1)):
        sc1 = Scattering1D(**common_kw, r_psi=r_psi, J=J, Q=Q1)
        sc2 = Scattering1D(**common_kw, r_psi=r_psi, J=J, Q=Q2)

        title = "r_psi={:.3g} | Q={} & {}, J={}".format(r_psi, Q1, Q2, J)
        viz_fb(sc1, sc2, title=title, savename=_savename(base_name, i))

    make_gif(**gif_common_kw, savepath=base_name + ".gif", duration=500,
             delimiter=base_name)

##############################################################################
# `r_psi` controls the redundancy of the filters (amount of overlap).
#
#   - Greater `r_psi` is greater temporal resolution, i.e. greater frequential
#     width. Hence, the largest scale, `J`, is attained later (in greater number
#     of wavelets). It also means that the *smallest* width wavelet is narrower,
#     hence the largest and smallest wavelets are farther in width and bandwidth.
#   - `r_psi` and `Q` aren't opposites or mutually cancelling. `Q` controls the
#     spacing between peaks, which results in different bandwidths to satisfy the
#     same overlap (`r_psi`), while `r_psi` doesn't affect spacing, only overlaps.
#   - To be precise, `r_psi` and `Q` each, differently, control the largest
#     bandwidth, but only `Q` controls the determination of subsequent CQT filters:
#     `next = func(Q) * previous`.
#   - Greater `r_psi` hence results in greater CQT portion.
#

##############################################################################
# Further reading
# ---------------
# Practical advice and extended description is found in documentation,
# `help(wavespin.Scattering1D)` and `help(wavespin.TimeFrequencyScattering1D)`.
