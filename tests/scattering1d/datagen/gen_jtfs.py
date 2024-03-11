# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Generate data for the TimeFrequencyScattering1D output tests.

Data is packed into a dictionary format: {string_key: numpy_array}

Data is read as a dictionary: out = loaded[string_key]

`out_type` is `'dict:list'` for debug convenience in `test_output()`; indices
to convert flattened arrays to individual joint slices are also provided.
"""
import os
import numpy as np
from copy import deepcopy
from wavespin import TimeFrequencyScattering1D

SAVEDIR = os.path.join('..', 'data', 'test_jtfs')
PRECISION = 'double'

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
        _, field, pair = k.split(':')
        if field not in meta_arr:
            meta_arr[field] = {}
        meta_arr[field][pair] = meta_flat[k]
    return meta_arr


def validate_meta_packing(meta_flat, meta):
    meta = deepcopy(meta)
    meta_arr = packed_meta_into_arr(meta_flat)

    for field in meta_arr:
        for pair in meta_arr[field]:
            o0 = meta_arr[field][pair]
            o1 = meta[field][pair]
            o0[np.isnan(o0)] = -2
            if isinstance(o1, list):
                o1 = np.array(o1)
            o1[np.isnan(o1)] = -2
            assert np.allclose(o0, o1)

#%%###########################################################################
# params should be such that jtfs.sc_freq.J_pad aren't all same
# `'dict:list'` for easier debugging in `test_outputs()`
common_params = dict(shape=1901, J=11, T=2**8, Q=16, J_fr=6, Q_fr=1,
                     pad_mode='reflect', pad_mode_fr='conj-reflect-zero',
                     max_pad_factor_fr=None, out_type='dict:list',
                     # 'primitive' is stable in case `smart_paths` algo changes
                     smart_paths='primitive', precision=PRECISION)
sfr0 = dict(sampling_filters_fr=('resample', 'resample'))
test_params = [
  dict(aligned=False, average_fr=True,  out_3D=False, F=32,
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
def pack_coeffs(out):
    """Can't save as `array(out[pair])` since jagged, and saving one by one
    completely flattened is very slow to read from, so save as flattened but
    one array and then "decode" by stored indices.
    """
    coeffs = {pair: [] for pair in list(out)}
    unpack_idxs = {f'{pair}_idxs': [] for pair in list(out)}

    for pair in out:
        pairi = pair + '_idxs'
        for i, c in enumerate(out[pair]):
            start = (unpack_idxs[pairi][-1][-1] if unpack_idxs[pairi] != [] else
                     0)
            end = start + c['coef'].shape[1]
            unpack_idxs[pairi].append((start, end))

        unpack_idxs[pairi] = np.array(unpack_idxs[pairi])
        coeffs[pair] = np.concatenate([c['coef'] for c in out[pair]],
                                      axis=1)
    return coeffs, unpack_idxs


def pack_meta(meta):
    meta_flat = {}
    for field in meta:
        if field != 'n':
            continue
        for pair in meta[field]:
            meta_flat["meta:{}:{}".format(field, pair)] = meta[field][pair]
    validate_meta_packing(meta_flat, meta)
    return meta_flat

def save(base_name, params, x, code, coeffs, unpack_idxs, meta_flat):
    _p = lambda name: os.path.join(SAVEDIR, base_name + name + '.npz')
    np.savez(_p('params'), **params, x=x, code=code)
    np.savez(_p('coeffs'), **coeffs, **unpack_idxs)
    np.savez(_p('meta'), **meta_flat)


for test_num in range(len(test_params)):
    params = {**test_params[test_num], **common_params}
    jtfs = TimeFrequencyScattering1D(**params, frontend='numpy')
    meta = jtfs.meta()

    out = jtfs(x)
    coeffs, unpack_idxs = pack_coeffs(out)
    meta_flat = pack_meta(meta)

    base_name = f"{test_num}_"
    save(base_name, params, x, code, coeffs, unpack_idxs, meta_flat)

#%%###########################################################################
# Note: implem no longer accounts for this case (# TODO clarify)
# but still worth keeping
# as it produces a unique structure
# TODO it dont reproduce the else path in core
# special case: `sc_freq.J_pad_fo > sc_freq.J_pad_max`, i.e. all first-order
# coeffs pad to greater than longest set of second-order, as in
# `U1 * phi_t * phi_f` and `(U1 * phi_t * psi_f) * phi_t * phi_f`.
# Note: due to various design changes (sigma0, num_intermediate, ...), this
# edge case is very hard to reproduce without using `max_noncqt_fr`.
params = dict(shape=2048, J=10, Q=8, J_fr=3, Q_fr=1, F=4, max_pad_factor=1,
              aligned=True, average_fr=True, out_type='dict:list', out_3D=True,
              max_pad_factor_fr=None, pad_mode_fr='zero', smart_paths='primitive',
              max_noncqt_fr=0, precision=PRECISION)
jtfs = TimeFrequencyScattering1D(**params, frontend='numpy')
meta = jtfs.meta()

# assert the edge case is reproduced
scf = jtfs.scf
assert scf._J_pad_fr_fo > scf.J_pad_frs_max_init, (
    scf._J_pad_fr_fo, scf.J_pad_frs_max_init, scf.N_frs_max, scf._n_psi1_f)

x = echirp(params['shape'])
out = jtfs(x)
coeffs, unpack_idxs = pack_coeffs(out)
meta_flat = pack_meta(meta)

base_name = f"{test_num + 1}_"
save(base_name, params, x, code, coeffs, unpack_idxs, meta_flat)
