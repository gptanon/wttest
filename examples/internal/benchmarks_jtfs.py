# -*- coding: utf-8 -*-
"""
Benchmarks - Time Scattering
============================
Speed profiling for the wavelet Time Scattering transform.

Tested on CPU and (if supported) GPU, with single precision.
"""
import torch
import torch.utils.benchmark as benchmark
from wavespin import TimeFrequencyScattering1D
from internal_utils import run_benchmarks, viz_benchmarks

got_gpu = bool(torch.cuda.is_available())

#%%###########################################################################
# Configure
# ---------
# 0 = long, 1 = short
CASE = 0
# number of trials of benchmarks to average times over
n_iters_cpu = 10
n_iters_gpu = n_iters_cpu * 10
# signal length
N = (2**17, 2**14)[CASE]
# transform params
J = (15, 12)[CASE]
Q = 16
J_fr = 5
Q_fr = 1
T = 2**(J - 2)
F = 2**J_fr

#%%###########################################################################
# Setups
# ------
x = torch.randn(N, dtype=torch.float32)
x_gpu = x.cuda()
ckw = dict(shape=N, J=J, Q=Q, T=T, J_fr=J_fr, Q_fr=Q_fr, F=F, average_fr=True,
           pad_mode='zero', max_pad_factor_fr=1, frontend='torch')

jtfs0 = TimeFrequencyScattering1D(**ckw)

#%% Gather & bench ###########################################################
bench_fns = {
    'WaveSpin': lambda: jtfs0(x),
}

#%%###########################################################################
# Run benchmarks & visualize
# --------------------------
# results_cpu = run_benchmarks(bench_fns, n_iters_cpu, verbose=True)

#%% GPU - setup ####################
if got_gpu:
    jtfs0.gpu()

    def wavespin_jtfs(x_gpu):
        jtfs0(x_gpu)

    t_gpu_fns = {
        'WaveSpin': benchmark.Timer(
            stmt='wavespin_jtfs(x_gpu)',
            setup='from __main__ import wavespin_jtfs',
            globals={'jtfs0': jtfs0, 'x_gpu': x_gpu},
            ),
    }

#%% GPU - execute ################
results_gpu = {}
if got_gpu:
    # warmup
    for name in t_gpu_fns:
        _ = t_gpu_fns[name].timeit(30)

    # main
    for name in t_gpu_fns:
        results_name = name + '-GPU'
        results_gpu[results_name] = t_gpu_fns[name].timeit(n_iters_gpu).mean
        print("%.5g" % results_gpu[results_name], results_name, flush=True)

#%% Organize data ############################################################
_results = {**results_cpu, **results_gpu}
preferred_order = ('WaveSpin-GPU', 'WaveSpin')
results = {name: _results[name] for name in preferred_order
           if name in _results}

#%% Visualize ################################################################
title = f"TimeFrequencyScattering1D: len(x)={N}"
viz_benchmarks(results, title)

#%% Local results ############################################################
"""
i7-7700HQ, GTX 1070

------------------------------------------------------------
len(x)=131072, Q=16, J=15, Q_fr=1, J_fr=5, T=2**J, F=2**J_fr
{
  'WaveSpin-GPU': 0.4805,

  'WaveSpin':     2.618,
}
-----------------------------------------------------------
len(x)=16384, Q=16, J=15, Q_fr=1, J_fr=5, T=2**J, F=2**J_fr
{
  'WaveSpin-GPU': 0.3375,

  'WaveSpin':     0.5441,
}
---------------------------------
"""
