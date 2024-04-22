# -*- coding: utf-8 -*-
"""Profiling `wavespin.utils.measures.algos`."""
import cProfile
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from timeit import Timer, default_timer as dtime
from functools import partial

from wavespin import Scattering1D
from wavespin.toolkit import pack_coeffs_jtfs, jtfs_to_numpy
from wavespin.utils.measures import compute_bandwidth, compute_bw_idxs
from wavespin.utils._compiled._algos  import (
    smallest_interval_over_threshold as
    smallest_interval_over_threshold_c,
    smallest_interval_over_threshold_indices as
    smallest_interval_over_threshold_indices_c
)
from wavespin.utils._compiled._algos2 import (
    smallest_interval_over_threshold2 as
    smallest_interval_over_threshold_c2,
    smallest_interval_over_threshold_indices2 as
    smallest_interval_over_threshold_indices_c2
)

def timeit(fn_partial, n_iters=50, n_repeats=20):
    _ = [fn_partial() for _ in range(5)]  # warmup
    return min(Timer(fn_partial).repeat(n_repeats, n_iters)) / n_iters

#%%
idx = 2
N = (128, 2048, 65536)[idx]
threshold = (9.9779875289978062, 160.7537193086604, 5147.719800470172)[idx]

pf0 = np.abs(Scattering1D(N).psi1_f[0][0])
pf = np.roll(pf0, -np.argmax(pf0) + len(pf0)//2 + 1)
pf = pf / np.max(pf)
pf = pf**2
c = np.argmax(pf)

fa0 = partial(smallest_interval_over_threshold_c,  pf, threshold, c)
fa1 = partial(smallest_interval_over_threshold_c2, pf, threshold, c)

interval0 = fa0()
interval1 = fa1()
interval_ref = compute_bandwidth(pf0)

assert interval0 == interval1 == interval_ref, (
    interval0, interval1, interval_ref)

args = (pf, threshold, c, interval_ref)
fb0 = partial(smallest_interval_over_threshold_indices_c,   *args)
fb1 = partial(smallest_interval_over_threshold_indices_c2,  *args)

idxs0 = fb0()
idxs1 = fb1()
idxs_ref = compute_bw_idxs(pf0)

cdiff = abs(np.argmax(pf0) - c)
idxs0 = (idxs0[0] - cdiff, idxs0[1] - cdiff - 1)
idxs1 = (idxs1[0] - cdiff, idxs1[1] - cdiff - 1)
assert idxs0 == idxs1 == idxs_ref, (
    idxs0, idxs1, idxs_ref)

#%%
n_iters = 100
n_repeats = 1000

# ta0 = timeit(fa0, n_iters, n_repeats)
# ta1 = timeit(fa1, n_iters, n_repeats)
# print('%.3g sec' % ta0)
# print('%.3g sec' % ta1); print()

tb0 = timeit(fb0, n_iters, n_repeats)
tb1 = timeit(fb1, n_iters, n_repeats)
print('%.3g sec' % tb0)
print('%.3g sec' % tb1)
