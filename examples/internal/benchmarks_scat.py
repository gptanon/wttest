# -*- coding: utf-8 -*-
"""
Benchmarks
==========
yes
"""
import torch
from wavespin import Scattering1D as WS1D
from kymatio import Scattering1D as KS1D
from internal_utils import run_benchmarks, viz_benchmarks

#%%###########################################################################
# Configure
# ---------
# device
device = ('cpu', 'gpu')[1]
# number of trials of benchmarks to average times over
n_iters = 10
# signal length
N = 2**15
# transform params
J = 14
Q = 16

#%%###########################################################################
# Setups
# ------
x = torch.randn(N, dtype=torch.float32)
ckw = dict(shape=N, J=J, Q=Q, frontend='torch')

sc0 = WS1D(**ckw)
sc1 = KS1D(**ckw)

if device == 'gpu':
    sc0.gpu()
    sc1.cuda()
    x = x.cuda()

#%% Gather & bench ###########################################################
bench_fns = {
    'WaveSpin': lambda: sc0(x),
    'Kymatio': lambda: sc1(x),
}

#%%###########################################################################
# Run benchmarks & visualize
# --------------------------
title = f"Scat1D: len(x)={N}"

results = run_benchmarks(bench_fns, n_iters, verbose=True)
viz_benchmarks(results, title)
