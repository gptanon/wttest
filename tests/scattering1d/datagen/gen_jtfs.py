# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Generate data for the TimeFrequencyScattering1D output tests."""
import numpy as np
from copy import deepcopy
from wavespin import TimeFrequencyScattering1D

SAVEDIR = '../data/'

def echirp(N, fmin=.1, fmax=None, tmin=0, tmax=1):
    fmax = fmax or N // 2
    t = np.linspace(tmin, tmax, N)

    a = (fmin**tmax / fmax**tmin) ** (1/(tmax - tmin))
    b = fmax**(1/tmax) * (1/a)**(1/tmax)

    phi = 2*np.pi * (a/np.log(b)) * (b**t - b**tmin)
    return np.cos(phi)


def packed_meta_into_arr(meta_flat):
    meta_arr = {}
    for k in meta_flat:  # rather, savefile
        if not k.startswith('meta:'):
            continue
        _, field, pair, i = k.split(':')
        if field not in meta_arr:
            meta_arr[field] = {}
        if pair not in meta_arr[field]:
            meta_arr[field][pair] = []
        meta_arr[field][pair].append(meta_flat[k])
    return meta_arr


def validate_meta_packing(meta_flat, meta):
    meta = deepcopy(meta)
    meta_arr = packed_meta_into_arr(meta_flat)

    for field in meta_arr:
        for pair in meta_arr[field]:
            meta_arr[field][pair] = np.array(meta_arr[field][pair])

            o0 = meta_arr[field][pair]
            o1 = meta[field][pair]
            o0[np.isnan(o0)] = -2
            if isinstance(o1, list):
                o1 = np.array(o1)
            o1[np.isnan(o1)] = -2
            assert np.allclose(o0, o1)

#%%###########################################################################
# params should be such that jtfs.sc_freq.J_pad aren't all same
common_params = dict(shape=1901, J=11, T=2**8, Q=16, J_fr=6, Q_fr=1,
                     pad_mode='reflect', pad_mode_fr='conj-reflect-zero',
                     max_pad_factor_fr=None, out_type='dict:list',
                     # 'primitive' is stable in case `smart_paths` algo changes
                     smart_paths='primitive')
sfr0 = dict(sampling_filters_fr=('resample', 'resample'))
test_params = [
  dict(aligned=False,  average_fr=True,  out_3D=False, F=32,
       max_pad_factor=None, sampling_filters_fr=('exclude', 'recalibrate')),
  dict(aligned=True,  average_fr=True,  out_3D=True,  F=4,  **sfr0),
  dict(aligned=False, average_fr=True,  out_3D=True,  F=16, **sfr0),
  dict(aligned=True,  average_fr=True,  out_3D=False, F='global', **sfr0),
  dict(aligned=True,  average_fr=False, out_3D=False, F=8,  **sfr0),
  dict(aligned=False, average_fr=True,  out_3D=True,  F=16,
       sampling_filters_fr=('recalibrate', 'recalibrate')),
]
for params in test_params:
    if 'max_pad_factor' not in params:
        params['max_pad_factor'] = 1

x = echirp(common_params['shape'])

# store generating source code
with open(__file__, 'r') as f:
    code = f.read()

#%%###########################################################################
def pack_coeffs(out, tp):
    packed = {}
    for pair in out:
        for i, c in enumerate(out[pair]):
            packed[pair + f':{i}'] = c['coef']
    return packed

def pack_meta(meta):
    meta_flat = {}
    for field in meta:
        if field != 'n':
            continue
        for pair in meta[field]:
            for i, v in enumerate(meta[field][pair]):
                meta_flat["meta:{}:{}:{}".format(field, pair, i)] = v
    validate_meta_packing(meta_flat, meta)
    return meta_flat

for test_num in range(len(test_params)):
    tp = test_params[test_num]
    params = {**tp, **common_params}
    jtfs = TimeFrequencyScattering1D(**params, frontend='numpy')
    meta = jtfs.meta()

    out = jtfs(x)
    coeffs = pack_coeffs(out, tp)
    meta_flat = pack_meta(meta)
    np.savez(SAVEDIR + f"test_jtfs_{test_num}.npz", **params, x=x,
             **coeffs, **meta_flat, code=code)

#%%###########################################################################
# Note: implem no longer accounts for this case but still worth keeping
# as it produces a unique structure
# special case: `sc_freq.J_pad_fo > sc_freq.J_pad_max`, i.e. all first-order
# coeffs pad to greater than longest set of second-order, as in
# `U1 * phi_t * phi_f` and `(U1 * phi_t * psi_f) * phi_t * phi_f`.
# Note: due to various design changes (sigma0, num_intermediate, ...), this
# edge case is very hard to reproduce without using `max_noncqt_fr`.
params = dict(shape=2048, J=10, Q=8, J_fr=3, Q_fr=1, F=4, max_pad_factor=1,
              aligned=True, average_fr=True, out_type='dict:list', out_3D=True,
              max_pad_factor_fr=None, pad_mode_fr='zero', smart_paths='primitive',
              max_noncqt_fr=0)
jtfs = TimeFrequencyScattering1D(**params, frontend='numpy')
meta = jtfs.meta()

# assert the edge case is reproduced
scf = jtfs.scf
assert scf._J_pad_fr_fo > scf.J_pad_frs_max_init, (
    scf._J_pad_fr_fo, scf.J_pad_frs_max_init, scf.N_frs_max, scf._n_psi1_f)

x = echirp(params['shape'])
out = jtfs(x)
tp = params
coeffs = pack_coeffs(out, tp)
meta_flat = pack_meta(meta)
np.savez(SAVEDIR + f"test_jtfs_{test_num + 1}.npz", **params, x=x, **coeffs,
         code=code)
