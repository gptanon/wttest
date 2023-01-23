# -*- coding: utf-8 -*-
"""
Benchmarks - Time Scattering
============================
Speed profiling for the wavelet Time Scattering transform.

Tested on CPU and (if supported) GPU, with single precision.
"""
import torch
import torch.utils.benchmark as benchmark
from wavespin import Scattering1D as WS1D
from kymatio import Scattering1D as KS1D
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
T = 2**J
Q = 16

#%%###########################################################################
# Setups
# ------
x = torch.randn(N, dtype=torch.float32)
x_gpu = x.cuda()
ckw = dict(shape=N, J=J, Q=Q, T=T, frontend='torch')

sc0 = WS1D(**ckw)
# sc1 = KS1D(**ckw)

#%% Make bench funcs #########################################################
bench_fns = {
    'WaveSpin': lambda: sc0(x),
    # 'Kymatio': lambda: sc1(x),
}

#%%###########################################################################
# Run benchmarks
# --------------
results_cpu = run_benchmarks(bench_fns, n_iters_cpu, verbose=True)

#%% GPU - setup ####################
if got_gpu:
    sc0.gpu()
    # sc1.cuda()

    def wavespin_sc(x_gpu):
        sc0(x_gpu)

    def kymatio_sc(x_gpu):
        sc1(x_gpu)

    t_gpu_fns = {
        'WaveSpin': benchmark.Timer(
            stmt='wavespin_sc(x_gpu)',
            setup='from __main__ import wavespin_sc',
            globals={'sc0': sc0, 'x_gpu': x_gpu},
            ),
        # 'Kymatio': benchmark.Timer(
        #     stmt='kymatio_sc(x_gpu)',
        #     setup='from __main__ import kymatio_sc',
        #     globals={'sc1': sc1, 'x_gpu': x_gpu},
        #     ),
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
preferred_order = ('WaveSpin-GPU', 'Kymatio-GPU', 'WaveSpin', 'Kymatio')
results = {name: _results[name] for name in preferred_order
           if name in _results}

#%% Visualize ################################################################
title = f"Scattering1D: len(x)={N}"
viz_benchmarks(results, title)

#%% Local results ############################################################
"""
i7-7700HQ, GTX 1070

---------------------------------
len(x)=131072, Q=16, J=15, T=2**J
{
  'WaveSpin-GPU': 0.05095,
  'Kymatio-GPU':  1.141,

  'WaveSpin':     0.9643,
  'Kymatio':      3.313,
}
--------------------------------
len(x)=16384, Q=16, J=12, T=2**J
{
  'WaveSpin-GPU': 0.03290,
  'Kymatio-GPU':  0.7120,

  'WaveSpin':     0.1405,
  'Kymatio':      0.8672,
}
--------------------------------
"""
