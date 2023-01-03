# -*- coding: utf-8 -*-
"""Reproduces most of the "Streaming JTFS" visual.

The outputs of this script were pieced together and edited with VideoPad.
"""
import os
import os.path as op
import warnings
from pathlib import Path
from copy import deepcopy
from timeit import default_timer as dtime

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Bbox

from wavespin import TimeFrequencyScattering1D
from wavespin.visuals import gif_jtfs_3d
from wavespin.toolkit import pack_coeffs_jtfs, jtfs_to_numpy


def no_border(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)


def split_subplots_in_two_vertically(fig, axes, spc=.01):
    # must have even number of rows
    n_rows = axes.shape[0]
    assert n_rows % 2 == 0, axes.shape

    # divide up into two sets of rows, top and bottom, for spacing
    # adjustment; keep top and bottom rows where they are
    row_idxs = list(range(n_rows))
    row_set_idxs = [i for i in row_idxs
                    if i != 0 and i != n_rows - 1]
    # per `i, row_set` logic to control spacing, the bottom rows are reverse-ordered
    row_sets = ([axes[i] for i in row_set_idxs[:len(row_set_idxs)//2]],
                [axes[i] for i in row_set_idxs[len(row_set_idxs)//2:][::-1]])

    for i, row_set in enumerate(row_sets):
        for j, row in enumerate(row_set):
            for ax in row:
                x0, y0, x1, y1 = ax.get_position().extents
                y0_new = y0 + spc*(j + 1)*(1 if i == 0 else -1)
                y1_new = y1 + spc*(j + 1)*(1 if i == 0 else -1)
                ax.set_position(Bbox(np.array([[x0, y0_new], [x1, y1_new]])))


class JTFS2DAnimator(animation.TimedAnimation):
    def __init__(self, jtfs, Scx, slice_size, step_size, w=1, h=1, preview=0,
                 spinned_only=1, cmx_scale=.8, cmap='turbo'):
        # process `Scx`
        # don't affect original input
        Scx = deepcopy(Scx)
        if not spinned_only:
            # equalize pairs
            # set all maxima to 1
            for pair in Scx:
                if '_up' not in pair and '_dn' not in pair:
                    Scx[pair] *= 1 / Scx[pair].max()
            # handle spinned separately, preserve assymetry
            # note choice of up is arbitrary and irrelevant
            up_max = Scx['psi_t * psi_f_up'].max()
            Scx['psi_t * psi_f_up'] /= up_max
            Scx['psi_t * psi_f_dn'] /= up_max
        else:
            cmx = max(Scx['psi_t * psi_f_up'].max(),
                      Scx['psi_t * psi_f_dn'].max())

        jmeta = jtfs.meta()
        Scx = pack_coeffs_jtfs(Scx, jmeta, structure=2, out_3D=jtfs.out_3D,
                               sampling_psi_fr=jtfs.sampling_psi_fr,
                               reverse_n1=False)
        # reverse psi_t ordering
        Scx = Scx[::-1]
        assert Scx.ndim == 4, Scx.shape

        self.Scx = Scx
        self.slice_size = slice_size
        self.step_size = step_size
        self.spinned_only = spinned_only

        self.n_frames = (Scx.shape[-1] - slice_size) // step_size + 1

        # configure args
        imshow_kw = dict(aspect='auto', cmap=cmap, animated=True,
                         interpolation='hanning')

        # get stuff
        n2s    = np.unique(jmeta['n']['psi_t * psi_f_up'][..., 0])
        n1_frs = np.unique(jmeta['n']['psi_t * psi_f_up'][..., 1])
        n_n2s, n_n1_frs = len(n2s), len(n1_frs)
        self.n_n2s, self.n_n1_frs = n_n2s, n_n1_frs
        # coeff max
        if not spinned_only:
            cmx = Scx.max()
        cmx *= cmx_scale

        # set up canvas
        n_rows = 2*n_n1_frs + 1
        n_cols = n_n2s + 1
        if spinned_only:
            n_rows -= 1
            n_cols -= 1

        width  = 11 * w
        height = 11 * n_rows / n_cols * h

        skw = dict(figsize=(width, height), dpi=72, facecolor='k')
        self.fig, self.axes = plt.subplots(n_rows, n_cols, **skw)

        # initialize image holders
        self.ims_up = []
        self.ims_dn = []
        if not spinned_only:
            self.ims_psi_t_phi_f = []
            self.ims_phi_t_psi_f = []
            self.ims_phi_t_phi_f = []

        # do plotting ########################################################
        def plot_spinned(up):
            for n2_idx in range(n_n2s):
                for n1_fr_idx in range(n_n1_frs):
                    # compute axis & coef indices ############################
                    if up:
                        row_idx = n1_fr_idx
                        coef_n1_fr_idx = n1_fr_idx
                    else:
                        row_idx = n1_fr_idx + 1 + n_n1_frs
                        coef_n1_fr_idx = n1_fr_idx + n_n1_frs + 1
                    col_idx = n2_idx + 1
                    coef_n2_idx = n2_idx + 1

                    if spinned_only:
                        col_idx -= 1
                        if not up:
                            row_idx -= 1

                    # visualize ##############################################
                    # coeffs
                    c = Scx[coef_n2_idx, coef_n1_fr_idx, :, :slice_size]
                    ax = self.axes[row_idx, col_idx]
                    im = ax.imshow(c, vmin=0, vmax=cmx, **imshow_kw)
                    # axis styling
                    no_border(ax)
                    # append
                    if up:
                        self.ims_up.append(im)
                    else:
                        self.ims_dn.append(im)

        plot_spinned(up=True)
        plot_spinned(up=False)

        # psi_t * phi_f ##########################################################
        if not spinned_only:
            row_idx = n_n1_frs
            coef_n1_fr_idx = n_n1_frs

            for n2_idx in range(n_n2s):
                # compute axis & coef indices
                col_idx = n2_idx + 1
                coef_n2_idx = n2_idx + 1

                # coeffs
                ax = self.axes[row_idx, col_idx]
                c = Scx[coef_n2_idx, coef_n1_fr_idx, :, :slice_size]
                im = ax.imshow(c, vmin=0, vmax=cmx, **imshow_kw)
                no_border(ax)
                self.ims_psi_t_phi_f.append(im)

        # phi_t * psi_f ##########################################################
        def plot_phi_t(up):
            col_idx = 0
            coef_n2_idx = 0
            for n1_fr_idx in range(n_n1_frs):
                if up:
                    row_idx = n1_fr_idx
                    coef_n1_fr_idx = n1_fr_idx
                else:
                    row_idx = n1_fr_idx + 1 + n_n1_frs
                    coef_n1_fr_idx = n1_fr_idx + 1 + n_n1_frs

                ax = self.axes[row_idx, col_idx]
                if not up:
                    c = Scx[coef_n2_idx, coef_n1_fr_idx, :, :slice_size]
                    # energy norm since we viz only once;
                    # did /= sqrt(2) in pack_coeffs_jtfs
                    c = c * np.sqrt(2)
                    im = ax.imshow(c, vmin=0, vmax=cmx, **imshow_kw)
                    self.ims_phi_t_psi_f.append(im)

                # axis styling
                no_border(ax)

        if not spinned_only:
            plot_phi_t(up=True)
            plot_phi_t(up=False)

        # phi_t * phi_f ######################################################
        if not spinned_only:
            row_idx = n_n1_frs
            col_idx = 0
            coef_n2_idx = 0
            coef_n1_fr_idx = n_n1_frs

            # coeffs
            c = Scx[coef_n2_idx, coef_n1_fr_idx, :, :slice_size]
            ax = self.axes[row_idx, col_idx]
            im = ax.imshow(c, vmin=0, vmax=cmx, **imshow_kw)
            # axis styling
            no_border(ax)
            self.ims_phi_t_phi_f.append(im)

        # finalize #######################################################
        subplots_adjust_kw = dict(left=.015, right=.985, bottom=.005, top=.995,
                                  wspace=.022, hspace=.035)
        self.fig.subplots_adjust(**subplots_adjust_kw)
        if spinned_only:
            split_subplots_in_two_vertically(self.fig, self.axes, .0027)
        if preview:
            plt.show()
        else:
            animation.TimedAnimation.__init__(self, self.fig, blit=True)

    def _draw_frame(self, frame_idx):
        i = frame_idx
        start = i * self.step_size
        end = start + self.slice_size
        t_slc = slice(start, end)

        # spinned
        for up in (True, False):
            im_idx = 0
            for n2_idx in range(self.n_n2s):
                for n1_fr_idx in range(self.n_n1_frs):
                    if up:
                        coef_n1_fr_idx = n1_fr_idx
                    else:
                        coef_n1_fr_idx = n1_fr_idx + self.n_n1_frs + 1
                    coef_n2_idx = n2_idx + 1

                    c = self.Scx[coef_n2_idx, coef_n1_fr_idx, :, t_slc]
                    if up:
                        self.ims_up[im_idx].set_array(c)
                    else:
                        self.ims_dn[im_idx].set_array(c)
                    im_idx += 1

        if not self.spinned_only:
            # psi_t * phi_f
            coef_n1_fr_idx = self.n_n1_frs
            for n2_idx in range(self.n_n2s):
                coef_n2_idx = n2_idx + 1
                c = self.Scx[coef_n2_idx, coef_n1_fr_idx, :, t_slc]
                self.ims_psi_t_phi_f[n2_idx].set_array(c)

            # phi_t * psi_f
            coef_n2_idx = 0
            for n1_fr_idx in range(self.n_n1_frs):
                coef_n1_fr_idx = n1_fr_idx + 1 + self.n_n1_frs
                c = self.Scx[coef_n2_idx, coef_n1_fr_idx, :, t_slc]
                c = c * np.sqrt(2)
                self.ims_phi_t_psi_f[n1_fr_idx].set_array(c)

            # phi_t * phi_f
            coef_n2_idx = 0
            coef_n1_fr_idx = self.n_n1_frs
            c = self.Scx[coef_n2_idx, coef_n1_fr_idx, :, t_slc]
            self.ims_phi_t_phi_f[0].set_array(c)

        # finalize
        self._drawn_artists = [*self.ims_up, *self.ims_dn]
        if not self.spinned_only:
            self._drawn_artists.extend([*self.ims_psi_t_phi_f,
                                        *self.ims_phi_t_psi_f,
                                        *self.ims_phi_t_phi_f])

    def new_frame_seq(self):
        return iter(range(self.n_frames))

    def _init_draw(self):
        pass

#%% Runtime checks ###########################################################
DATADIR = 'data'
OUTDIR = 'out'

DATADIR = str(Path(Path(__file__).parent, DATADIR).resolve())
OUTDIR = str(Path(Path(__file__).parent, OUTDIR).resolve())
if not os.path.isdir(DATADIR) or len(os.listdir(DATADIR)) == 0:
    raise RuntimeError("`DATADIR` is empty or doesn't exist. Copy files from "
                       "wavespin/examples/internal/data/")
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

#%% Configure ################################################################
CASE = 0
GPU = bool(torch.cuda.is_available())

#%%
path = {
  0: op.join(DATADIR, 'pre_pad_librosa_trumpet_slow.npy'),
  1: op.join(DATADIR, 'shepard_tone.npy'),
  2: op.join(DATADIR, 'brain_waves.npy'),
}[CASE]

common_kw = dict(
    F=4, Q_fr=1, J_fr=3, Q=12,
    out_type='dict:array', out_3D=1,
    average_fr=1, F_kind='decimate',
    pad_mode='zero',  # assume silence before audio
    frontend='torch', max_pad_factor_fr=1,
    precision='single', max_noncqt_fr=0,
)
_default = (-4, -4)
J_minus_log2_N = {
  0: (-6, -4),
  1: (-6, -4),
  2: _default,
}[CASE]
paths_exclude = {
  0: {'n2': [2, 3, 4, 5, 6, 7, 13], 'n1_fr': [3, 4]},
  1: {'n2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17], 'n1_fr': [3, 4]},
  2: {'n2': [2, 3, 4, 5, 6, 7, 12], 'n1_fr': [-1, -2]},
}[CASE]
# anim slice size, t_end = t_start + slice_size -- data[..., t_start:t_end]
slice_size = {
  0: 75,
  1: 75,
  2: 50,
}[CASE]
# anim step size, t_start_next = t_start_current + step_size
step_size = {
  0: 3,
  1: 4,
  2: 3,
}[CASE]
# frames per sec
_default = 25
fps = {
  0: 27,
  1: _default,
  2: _default,
}[CASE]
# target num of frames
# result will be between this and its nextpow2, e.g. `256` is up to `511`
_default = 256
target_n_frames = {
  0: _default,
  1: 512,
  2: _default,
}[CASE]
# color map
_default = 'turbo'
cmap = {
  0: _default,
  1: 'seismic',
  2: 'twilight',
}[CASE]
# color norm
_default = .75
cmx_scale = {
  0: _default,
  1: _default,
  2: _default,
}[CASE]

#%% Execute ##################################################################
# load `x`
x = np.load(path)

#%% make JTFS & scatter ------------------------------------------------------
N = len(x)
# for sake of adapting J to N, pick closest
log2_N = int(round(np.log2(N)))
J = (log2_N + J_minus_log2_N[0],
     log2_N + J_minus_log2_N[1])
# achieve TARGET_N_FRAMES within next power of 2
T = 2**int(np.floor(np.log2(len(x) / target_n_frames)))
# avoid excess compute
max_pad_factor = 0 if N >= 262144 else 1

cfg = dict(shape=N, J=J, T=T, paths_exclude=paths_exclude,
           max_pad_factor=max_pad_factor, **common_kw)
jtfs = TimeFrequencyScattering1D(**cfg)
if GPU:
    jtfs.gpu()
Scxo = Scx = jtfs_to_numpy(jtfs(x))

print(Scxo['psi_t * psi_f_up'].shape)

#%% Visualize ################################################################
preview = 0
spinned_only = 1
ext = ('.gif', '.mp4')[1]
name = Path(path).name[:6] + 'X' + cmap + ext
savepath = op.join(OUTDIR, name)

t0 = dtime()
ani = JTFS2DAnimator(jtfs, Scx, slice_size=slice_size, step_size=step_size,
                     preview=preview, spinned_only=spinned_only,
                     cmx_scale=cmx_scale, cmap=cmap)
if not preview:
    writer = 'ffmpeg' if savepath.endswith('.gif') else None
    ani.save(savepath, fps=fps, savefig_kwargs=dict(pad_inches=0), writer=writer)
    plt.close()
    print("%.3g sec to animate" % (dtime() - t0))

#%% 3D visual #################################################################
if CASE != 0:
    # may have used r_psi>.9 for other viz; other changes
    warnings.warn("Only `CASE==0` is fully reproduced for the 3D visual. "
                  "Other cases are approximately reproduced with 0's config.")
if CASE == 0:
    x = np.load(op.join(DATADIR, 'librosa_trumpet_slow.npy'))

N = len(x)
# for sake of adapting J to N, pick closest
log2_N = int(round(np.log2(N)))
if CASE == 0:
    # ended up slightly increasing len(x) and changing anim params, adj manually
    log2_N -= 1
T = 2**(log2_N - 7)

common_kw = dict(
    F=4, Q_fr=2, J_fr=4, Q=(16, 1),
    out_type='dict:array', out_3D=1,
    average_fr=1, frontend='torch',
    pad_mode='zero',  # assume silence before audio
    max_pad_factor=None, max_pad_factor_fr=None,
    precision='single', sampling_filters_fr='resample',
)
if CASE != 0:
    common_kw['F_kind'] = 'decimate'
_default = log2_N - 3
J = {
  0: (11, 13),
  1: _default,
  2: _default,
}[CASE]

#%% handle cmap unsupported by plotly
if cmap in ('seismic', 'twilight'):
    # borrowed from https://plotly.com/python/v3/matplotlib-colorscales/
    import matplotlib as mpl

    cmap_cmap = mpl.cm.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=0, vmax=255)

    pl_entries = 255
    h = 1.0/(pl_entries-1)
    cmap_pl = []
    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap_cmap(k*h)[:3])*255))
        cmap_pl.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
else:
    cmap_pl = cmap

#%% Build scattering object
cfg = dict(shape=N, J=J, T=T, **common_kw)
jtfs = TimeFrequencyScattering1D(**cfg)
if GPU:
    torch.cuda.empty_cache()
    jtfs.gpu()

#%% Compute
Scx = Scxo = jtfs_to_numpy(jtfs(x))

print(Scxo['psi_t * psi_f_up'].shape)

#%% Make visual
t0 = dtime()
savedir = op.join(OUTDIR, 'librosa_trumpet_3d6')
if not op.isdir(savedir):
    os.mkdir(savedir)
gif_jtfs_3d(Scx, jtfs, preset='spinned', savedir=savedir,
            base_name='im', overwrite=True, cmap=cmap_pl,
            gif_kw=dict(duration=28), save_images=1, cmap_norm=.65,
            angles='rotate')
print(dtime() - t0)

#%%
template = (savedir + os.sep).replace(r'\\', '/') + 'im%03d.png'
savepath = op.join(savedir, 'out.avi').replace(r'\\', '/')
cmd = (rf'ffmpeg -framerate 43 -y -i "{template}" -strict -2 -an -b:v 32M '
       rf'"{savepath}"')
os.system(cmd)
