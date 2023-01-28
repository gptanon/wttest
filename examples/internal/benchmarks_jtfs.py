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
# 0 = averaged, 1 = unaveraged
CFG = 0
# number of trials of benchmarks to average times over
n_iters_cpu = 100
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
average_fr = (True, False)[CFG]
out_3D     = (True, False)[CFG]

#%%###########################################################################
# Setups
# ------
x = torch.randn(N, dtype=torch.float32)
x_gpu = x.cuda()
ckw = dict(shape=N, J=J, Q=Q, T=T, J_fr=J_fr, Q_fr=Q_fr, F=F, out_3D=out_3D,
           average_fr=average_fr, pad_mode='zero', max_pad_factor_fr=1,
           vectorized_fr=2,
           frontend='torch')

jtfs0 = TimeFrequencyScattering1D(**ckw)

#%% Gather & bench ###########################################################
bench_fns = {
    'WaveSpin': lambda: jtfs0(x),
}

#%%###########################################################################
# Run benchmarks & visualize
# --------------------------
results_cpu = run_benchmarks(bench_fns, n_iters_cpu, verbose=True)

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
title = "TimeFrequencyScattering1D: len(x)={}, out_3D={}".format(
    N, out_3D)
viz_benchmarks(results, title)

#%% Local results ############################################################
"""
i7-7700HQ, GTX 1070

 - COMMONS: Q=16, J=15, Q_fr=1, J_fr=5, T=2**(J - 2), F=2**J_fr
 - `average_fr=out_3D`
"""
results_all = {
  (131072, True):
    {
      'WaveSpin-GPU': 0.1788,
      'OldSpin-GPU':  0.5209,

      'WaveSpin':     5.345,
      'OldSpin':      5.385,
    },
  (131072, False):
    {
      'WaveSpin-GPU': 0.07413,
      'OldSpin-GPU':  0.4143,

      'WaveSpin':     2.082,
      'OldSpin':      2.476,
    },
  (16384, True):
    {
      'WaveSpin-GPU': 0.0642,
      'OldSpin-GPU':  0.3232,

      'WaveSpin':     0.6307,
      'OldSpin':      0.8214,
    },
  (16384, False):
    {
      'WaveSpin-GPU': 0.04789,
      'OldSpin-GPU':  0.2660,

      'WaveSpin':     0.2524,
      'OldSpin':      0.4363,
    },
}
if 0:
    preferred_order = [(131072, True), (16384, True),
                       (131072, False), (16384, False)]
    results_all_ordered = {k: results_all[k] for k in preferred_order}
    title_template = "JTFS: len(x)={}, out_3D={}"
    viz_benchmarks(results_all_ordered, title_template, nested=True)
