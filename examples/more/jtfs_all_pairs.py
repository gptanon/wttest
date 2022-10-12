# -*- coding: utf-8 -*-
"""
JTFS All Pairs Case
===================
Visualize JTFS of a signal with time-frequency geometries that maximize each pair.

We take exponential chirp + pure sine + impulse, also show scalogram, and
2D time-frequency wavelets for the corresponding JTFS coefficients.
Example is designed to "look nice", but practical scenarios still realize intent.
"""
import numpy as np
from wavespin.numpy import TimeFrequencyScattering1D, Scattering1D
from wavespin.toolkit import echirp
from wavespin.visuals import viz_jtfs_2d, scalogram
from wavespin import CFG

# sampling rate - only affects axis labels
#  None: will label axes with discrete units (cycles/sample), `xi`
#  int: will label axes with physical units (Hz), `xi *= Fs`
FS = 4096

# TODO jtfs smart paths .0075 cause 2nd order matter more
# TODO N_fr_p2up=True default cause improve quality of coeffs
# TODO undo bs

#%% Generate echirp and create scattering object #############################
N = 4096
# span low to Nyquist; assume duration of 1 second
x = echirp(N, fmin=64, fmax=N/2) / 2

x += np.cos(2*np.pi * 364 * np.linspace(0, 1, N)) / 2
x[N//2-16:N//2+16] += 5

#%% Build scattering objects #################################################
CFG['JTFS']['N_fr_p2up'] = True

ckw = dict(shape=N, J=(8, 8), Q=(16, 1), T=2**8)
jtfs = TimeFrequencyScattering1D(**ckw, J_fr=3, Q_fr=1,
                                 sampling_filters_fr='resample', analytic=1,
                                 average=0, average_fr=0, F=8,
                                 out_type='dict:list',
                                 pad_mode_fr=('conj-reflect-zero', 'zero')[1],
                                 oversampling=999, oversampling_fr=999,
                                 # drop some coeffs to avoid excessive plot size
                                 paths_exclude={'n2':    [2, 3, 4, -1],
                                                'n1_fr': [-1]},
                                 max_pad_factor=None, max_pad_factor_fr=None,
                                 max_noncqt_fr=0, smart_paths=.0075)
sc = Scattering1D(**ckw, average=False, out_type='list')

#%% Show scalogram ###########################################################
# don't need to configure all this; it's only here to reproduce
# what's shown for the paper
plot_cfg = {
  'title_x': "Exponential chirp + pure sine + pulse",
  'title_scalogram': "Scalogram",
  'label_kw_xy': dict(weight='bold', fontsize=14),
  'title_kw':    dict(weight='bold', fontsize=16),
  'tick_params': dict(labelsize=14),
}
scalogram(x, sc, show_x=True, fs=FS, plot_cfg=plot_cfg)

#%% Take JTFS ################################################################
Scx_orig = jtfs(x)
for pair in Scx_orig:
    if pair.startswith('phi_t'):
        for c in Scx_orig[pair]:
            c['coef'] *= 0

#%% Visualize JTFS ###########################################################
vkw = dict(
    viz_filterbank=0,
    viz_coeffs=1,
    viz_spins=(1, 1),
    axis_labels=1,
    w=1.,
    h=1.,
    show=1,
    savename='jtfs_viz_2d',
)
# don't need to configure all this; it's only here to reproduce the visual
plot_cfg = {
  'phi_t_blank': True,
  'phi_t_loc': 'bottom',
  'label_kw_xy':   dict(fontsize=20),
  'title_kw':      dict(weight='bold', fontsize=26, y=1.02),
  'suplabel_kw_x': dict(weight='bold', fontsize=24, y=-.037),
  'suplabel_kw_y': dict(weight='bold', fontsize=24, x=-.066),
  'imshow_kw_filterbank': dict(aspect='auto', cmap='bwr'),
  'imshow_kw_coeffs':     dict(aspect='auto', cmap='turbo'),
  'subplots_adjust_kw': dict(left=0, right=1, bottom=0, top=1,
                             wspace=.02, hspace=.02),
  'savefig_kw': dict(bbox_inches='tight'),
  'filterbank_zoom': .9,
  'coeff_color_max_mult': .8,
  'filter_part': 'real',
}

viz_jtfs_2d(jtfs, Scx_orig, fs=FS, psi_id=0, plot_cfg=plot_cfg, **vkw)
