# -*- coding: utf-8 -*-
"""Profiling `wavespin.toolkit.pack_coeffs_jtfs`."""
import cProfile
import numpy as np
import torch
from timeit import Timer, default_timer as dtime
from functools import partial

from wavespin import TimeFrequencyScattering1D
from wavespin.toolkit import pack_coeffs_jtfs, jtfs_to_numpy

USE_TIMEIT = 1

#%%
def timeit(fn_partial, n_iters=50, n_repeats=5):
    return min(Timer(fn_partial).repeat(n_repeats, n_iters)) / n_iters

#%%
def setup():
    GPU = 1
    is_torch = 1
    N = 8192
    T = N // 32
    J_fr = 7
    out_type = ('dict:list', 'dict:array')[0]

    J = int(np.log2(N))

    x = np.random.randn(N)
    x = np.vstack([x]*4)
    if is_torch:
        x = torch.from_numpy(x)
    if GPU:
        x = x.cuda()

    objs_all = []
    for out_3D in (True, False):
        average_fr = bool(out_3D)
        jtfs = TimeFrequencyScattering1D(
            N, out_type=out_type, T=T, J=J, J_fr=J_fr,
            out_3D=out_3D, average_fr=average_fr,
            frontend='torch' if is_torch else 'numpy')
        jkw = dict(out_3D=jtfs.scf.out_3D,
                   sampling_psi_fr=jtfs.scf.sampling_psi_fr)
        if GPU:
            jtfs.gpu()

        jmeta = jtfs.meta()
        Scx = jtfs(x)
        objs_all.append((jtfs, jmeta, jkw, Scx))
    return objs_all, is_torch


def main0(objs_all, is_torch):
    total = 0
    # 18 total loops
    for objs in objs_all:
        jtfs, jmeta, jkw, Scx = objs
        for structure in (1, 2, 3, 4, 5):
            for separate_lowpass in (False, True):
                if structure == 5 and separate_lowpass:
                    continue  # invalid

                fn_partial = partial(pack_coeffs_jtfs,
                                     Scx, jmeta, structure=structure,
                                     separate_lowpass=separate_lowpass,
                                     # as_numpy=is_torch,
                                     **jkw)
                total += timeit(fn_partial)
        # GPU: this may be ill-placed; may be better to `torch.utils.benchmark`
        # without it, or (once over all loops) `timeit.default_timer` with it
        if is_torch:
            torch.cuda.synchronize()
    print(total / 18)


def main1(objs_all, is_torch):
    for objs in objs_all:
        jtfs, jmeta, jkw, Scx = objs
        for structure in (1, 2, 3, 4, 5):
            for separate_lowpass in (False, True):
                if structure == 5 and separate_lowpass:
                    continue  # invalid

                if is_torch:
                    # not needed for non-GPU
                    # note: `as_numpy=1` doesn't account for `dict:array`
                    # overhead, but that overhead's small on GPU (maybe not
                    # for all configs, but likely still a net-positive)
                    def pack_coeffs_jtfs_torch_synch():
                        out = pack_coeffs_jtfs(
                            Scx, jmeta,
                            structure=structure,
                            separate_lowpass=separate_lowpass,
                            # as_numpy=1,
                            **jkw)
                        torch.cuda.synchronize()
                        return out

                    for _ in range(250):
                        _ = pack_coeffs_jtfs_torch_synch()
                else:
                    for _ in range(250):
                        _ = pack_coeffs_jtfs(
                            Scx, jmeta,
                            structure=structure,
                            separate_lowpass=separate_lowpass,
                            **jkw)

#%%
args = setup()
if USE_TIMEIT:
    main0(*args)
else:
    cProfile.run('main1(*args)', 'profile.prof')

#%%
# note: values might be outdated
"""
NPY
.0064

TORCH-CPU
.024

TORCH-GPU
.019

"""