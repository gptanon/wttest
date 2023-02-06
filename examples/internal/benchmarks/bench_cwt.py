# -*- coding: utf-8 -*-
"""
Benchmarks - Continuous Wavelet Transform
=========================================
Speed profiling for the Continuous Wavelet Transform with stride support, for:

    - WaveSpin
    - ssqueezepy
    - MATLAB
    - PyWavelets
    - scipy

Tested on CPU and (if supported) GPU, with single precision.

See README in `examples/internals/benchmarks` on design of this benchmark.
"""
import os
import numpy as np
import pywt
import scipy
import ssqueezepy as ssq
import torch.utils.benchmark as benchmark

from wavespin import Scattering1D
from benchmarks_utils import run_benchmarks, viz_benchmarks

#%%############################################################################
# Determine GPU availability
# --------------------------
got_gpu = {lib: False for lib in ('WaveSpin', 'ssqueezepy')}
try:
    import torch
    if torch.cuda.is_available():
        got_gpu['WaveSpin'] = True
except ImportError:
    pass
try:
    import cupy
    if cupy.cuda.runtime.getDeviceCount() > 0:
        got_gpu['ssqueezepy'] = True
except ImportError:
    pass

#%%###########################################################################
# Configure
# ---------
# 0 = long, 1 = short
CASE = 1
# number of trials of benchmarks to average times over
n_iters_cpu = 100
n_iters_gpu = n_iters_cpu * 10
# signal length
N = (2**17, 2**14)[CASE]
# stride / hop
hop_size = (16, 4)[CASE]

#%%###########################################################################
# Setups
# ------
# Note pywt and scipy can't store filters, so there's no object creation setup.
x = np.random.randn(N).astype('float32')
lib_args = {}

#%% WaveSpin #################################################################
J = (15, 12)[CASE]
# with CWT we'll have higher redundancy than with scattering, hence `r_psi`.
# ~240 wavelets is a practical ballpark.
# Note, with same quality factor for both `CASE`, a different `hop_size` isn't
# warranted, but matching QF to practical `x` makes no difference for benching
scat_cfg = dict(shape=N, J=J, Q=19, r_psi=.8, max_pad_factor=1,
                pad_mode='zero', precision='single')
sc = Scattering1D(**scat_cfg)
n_psis = len(sc.psi1_f)
N_psi = len(sc.psi1_f[0][0])
lib_args['WaveSpin'] = dict(x=x, hop_size=hop_size)

# use to determine scales start and end for scale-based implementations;
# for best fairness we set it based on largest temporal support, as some implems
# (scipy, pywt) sample in time and are pure CQT as opposed to CQT + ~STFT -
# matching freqs doesn't matter for performance
fmax = sc.psi1_f[0]['xi']
fmin = [p['xi'] for p in sc.psi1_f if p['is_cqt']][-1]
fmax_fmin_ratio = fmax / fmin
# quality factor
QF = sc.psi1_f[0]['xi'] / sc.psi1_f[0]['sigma']

#%% ssqueezepy ###############################################################
ssq_wav_cfg = dict(N=N_psi, wavelet=('morlet', {'mu': QF}), dtype='float32')
ssq_wav = ssq.Wavelet(**ssq_wav_cfg)
# `20 / np.pi` places peak at Nyquist, as ssq wavelet arg spans up to `pi`
s_min = ssq_wav.config['mu'] / np.pi * (0.5 / fmax)
s_max = s_min * fmax_fmin_ratio
# we can match exactly but no need, simplify by treating non-CQT as CQT
scales_dummy = np.logspace(np.log10(s_min), np.log10(s_max), n_psis)
_ = ssq_wav.Psih(scale=scales_dummy)
lib_args['ssqueezepy'] = dict(x=x, wavelet=ssq_wav, scales=scales_dummy,
                              padtype='zero')

#%% scipy ####################################################################
# we omit derivations, but one can confirm ~exact match in unaliased regime
# ("~" per scipy lacking the Gabor mean-correcting term)
def s_w_from_xi_sigma(xi, sigma):
    s = 1 / (sigma * 2*np.pi)
    w = xi / sigma
    return s, w

pf_cqt    = [pf for pf in sc.psi1_f if     pf['is_cqt']]
pf_noncqt = [pf for pf in sc.psi1_f if not pf['is_cqt']]
pf_cqt_fmax = pf_cqt[0]
pf_noncqt_fmin = pf_noncqt[-1]

xi_max_cqt,    sigma_max_cqt    = [pf_cqt_fmax[nm]    for nm in ('xi', 'sigma')]
xi_min_noncqt, sigma_min_noncqt = [pf_noncqt_fmin[nm] for nm in ('xi', 'sigma')]
s_min_cqt, w_cqt = s_w_from_xi_sigma(xi_max_cqt, sigma_max_cqt)
s_max_noncqt, _ = s_w_from_xi_sigma(xi_min_noncqt, sigma_min_noncqt)

sci_widths = np.logspace(np.log10(s_min_cqt), np.log10(s_max_noncqt), n_psis)
sci_wav = lambda *a, **k: scipy.signal.morlet2(*a, **k, w=w_cqt)
lib_args['scipy'] = dict(data=x, wavelet=sci_wav, widths=sci_widths)

#%% PyWavelets ###############################################################
# pywt's wavelet generation makes it impossible to replicate arbitrary
# filterbanks, and approximation is difficult to decode, so we build something
# close enough, unfortunately forcing hard-coding here, to `QF ~ 37`.
# Note, this designs for integrated wavelets, but CWT's output is then
# differentiated; the difference is small and negligible for our purposes.
pywt_wav = 'cmor10.0-2.7'
pywt_scales = np.logspace(np.log10(5.5), np.log10(5.5*fmax_fmin_ratio), n_psis)
# pywt's default `method='conv'` is *much* slower
lib_args['PyWavelets'] = dict(data=x, wavelet=pywt_wav, scales=pywt_scales,
                              method='fft')

#%% WaveSpin-GPU ###########################################################
if got_gpu['WaveSpin']:
    # could've initialized torch from start but cpu main use is numpy
    sc_gpu = Scattering1D(**scat_cfg, frontend='torch')
    sc_gpu.gpu()
    x_gpu = torch.as_tensor(x, dtype=torch.float32, device='cuda')

#%% ssqueezepy-GPU ###########################################################
if got_gpu['ssqueezepy']:
    os.environ['SSQ_GPU'] = '1'
    ssq_wav_gpu = ssq.Wavelet(**ssq_wav_cfg)
    scales_dummy_gpu = torch.as_tensor(scales_dummy, dtype=torch.float32,
                                       device='cuda')
    _ = ssq_wav_gpu.Psih(scale=scales_dummy_gpu)
    os.environ['SSQ_GPU'] = '0'  # put back on CPU for now

#%% Make bench funcs (CPU) ###################################################

# in case we want to bench with padding (pywt & scipy effectively only zero-pad)
# def cwt_with_pad_and_unpad(cwt_fn, args):
#     # need to pad and unpad manually if the library doesn't provide it
#     x = args['data']
#     pad_right = (N_psi - len(x)) // 2
#     pad_left = N_psi - pad_right - len(x)
#     args['data'] = np.pad(x, [pad_left, pad_right])

#     out = cwt_fn(**args)
#     if isinstance(out, tuple):
#         out = out[0]
#     out = out[..., pad_left:-pad_right]
#     # restore original to dict
#     args['data'] = x
#     return out

# cwt & bench functions
def cwt_wavespin(**args):
    _ = sc.cwt(**args)

def cwt_ssqueezepy(**args):
    out = ssq.cwt(**args)[0]
    out = out[..., ::hop_size]

def cwt_scipy(**args):
    # out = cwt_with_pad_and_unpad(scipy.signal.cwt, args)
    out = scipy.signal.cwt(**args)
    out = out[..., ::hop_size]

def cwt_pywavelets(**args):
    # out = cwt_with_pad_and_unpad(pywt.cwt, args)
    out = pywt.cwt(**args)[0]
    out = out[..., ::hop_size]

cwt_fns = {
    'WaveSpin': cwt_wavespin,
    'ssqueezepy': cwt_ssqueezepy,
    'PyWavelets': cwt_pywavelets,
    'scipy': cwt_scipy,
}
bench_fns = {name: lambda name=name: cwt_fns[name](**lib_args[name])
             for name in lib_args}

#%% Make bench funcs (GPU ####################################################
def cwt_wavespin_gpu(x_gpu, hop_size):
    _ = sc_gpu.cwt(x_gpu, hop_size)

def cwt_ssqueezepy_gpu(x_gpu, ssq_wav_gpu, scales_dummy_gpu, hop_size):
    out = ssq.cwt(x_gpu, ssq_wav_gpu, scales_dummy_gpu)[0]
    out = out[..., ::hop_size]

t_gpu_fns = {}
if got_gpu['WaveSpin']:
    t_gpu_fns['WaveSpin'] = benchmark.Timer(
        stmt='cwt_wavespin_gpu(x_gpu, hop_size)',
        setup='from __main__ import cwt_wavespin_gpu',
        globals={'sc_gpu': sc_gpu, 'x_gpu': x_gpu, 'hop_size': hop_size},
    )
if got_gpu['ssqueezepy']:
    t_gpu_fns['ssqueezepy'] = benchmark.Timer(
        stmt='cwt_ssqueezepy_gpu(x_gpu, ssq_wav_gpu, scales_dummy_gpu, hop_size)',
        setup='from __main__ import cwt_ssqueezepy_gpu',
        globals={'sc_gpu': sc_gpu, 'x_gpu': x_gpu, 'ssq_wav_gpu': ssq_wav_gpu,
                 'scales_dummy_gpu': scales_dummy_gpu, 'hop_size': hop_size},
    )

#%%###########################################################################
# Run benchmarks
# --------------
os.environ['SSQ_GPU'] = '0'
results_cpu = run_benchmarks(bench_fns, n_iters_cpu, verbose=True)

#%% now GPU ##################
os.environ['SSQ_GPU'] = '1'

# warmup
for name in t_gpu_fns:
    _ = t_gpu_fns[name].timeit(3)

# main
results_gpu = {}
for name in t_gpu_fns:
    results_name = name + '-GPU'
    results_gpu[results_name] = t_gpu_fns[name].timeit(n_iters_gpu).mean
    print("%.5g" % results_gpu[results_name], results_name, flush=True)

#%%###########################################################################
# Organize data
# -------------
results_matlab = {
  0: {'MATLAB': 1.532,
      'MATLAB-GPU': 0.1251},
  1: {'MATLAB': 0.1261,
      'MATLAB-GPU': 0.0603},
}
_results = {**results_cpu, **results_gpu, **results_matlab}
preferred_order = ('WaveSpin-GPU', 'ssqueezepy-GPU', 'MATLAB-GPU',
                   'WaveSpin', 'ssqueezepy', 'MATLAB', 'PyWavelets', 'scipy')
results = {name: _results[name] for name in preferred_order
           if name in _results}

#%%###########################################################################
# Visualize
# ---------
title = f"CWT: len(x)={N}, hop_size={hop_size}"
viz_benchmarks(results, title)

#%% Local results ############################################################
"""
i7-7700HQ, GTX 1070

----------------------------
len(x)=131072, hop_size=16
{
  'WaveSpin-GPU':    0.01104,
  'ssqueezepy-GPU':  0.02784,
  'MATLAB-GPU':      0.1251,

  'WaveSpin':        0.3588,
  'ssqueezepy':      0.9758,
  'MATLAB':          1.532,
  'PyWavelets':      3.542,
  'scipy':           5.664,
}
----------------------------
len(x)=16384, hop_size=4
{
  'WaveSpin-GPU':    0.001191,
  'ssqueezepy-GPU':  0.004126,
  'MATLAB-GPU':      0.01079,

  'WaveSpin':        0.05243,
  'ssqueezepy':      0.07476,
  'MATLAB':          0.1261,
  'PyWavelets':      0.2837,
  'scipy':           0.4871,
}
----------------------------
"""
