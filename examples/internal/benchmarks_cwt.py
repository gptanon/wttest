# -*- coding: utf-8 -*-
"""
Benchmarks
==========
yes
"""
import numpy as np
import pywt
import scipy
import ssqueezepy as ssq

from wavespin import Scattering1D
from internal_utils import run_benchmarks, viz_benchmarks

#%%###########################################################################
# Configure
# ---------
# number of trials of benchmarks to average times over
n_iters = 10
# signal length  # TODO do two or tell users to swap manually
N = 2**14
# stride, hop
hop_size = 8

#%%###########################################################################
# Setups
# ------
# Note pywt and scipy can't store filters, so there's no object creation setup.
x = np.random.randn(N).astype('float32')
lib_args = {}

#%% WaveSpin #################################################################
sc = Scattering1D(shape=N, Q=16, J=12, max_pad_factor=1, precision='single')
n_psis = len(sc.psi1_f)
N_psi = len(sc.psi1_f[0][0])
lib_args['WaveSpin'] = dict(x=x, hop_size=hop_size)

# use to determine scales start and end for scale-based implementations
fmax = sc.psi1_f[0]['xi']
fmin = sc.psi1_f[-1]['xi']
fmax_fmin_ratio = fmax / fmin
# quality factor
QF = sc.psi1_f[0]['xi'] / sc.psi1_f[0]['sigma']

#%% ssqueezepy ###############################################################
ssq_wav = ssq.Wavelet(N=N_psi, wavelet=('morlet', {'mu': QF}), dtype='float32')
# `20 / np.pi` places peak at Nyquist, as ssq wavelet arg spans up to `pi`
s_min = ssq_wav.config['mu'] / np.pi * (0.5 / fmax)
s_max = s_min * fmax_fmin_ratio
# we can match exactly but no need, simplify by treating non-CQT as CQT
scales_dummy = np.logspace(np.log10(s_min), np.log10(s_max), n_psis)
_ = ssq_wav.Psih(scale=scales_dummy)
lib_args['ssqueezepy'] = dict(x=x, wavelet=ssq_wav, scales=scales_dummy)

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

xi_max_cqt, sigma_max_cqt = [pf_cqt_fmax[nm] for nm in ('xi', 'sigma')]
xi_min_noncqt, sigma_min_noncqt = [pf_noncqt_fmin[nm] for nm in ('xi', 'sigma')]
s_min_cqt, w_cqt = s_w_from_xi_sigma(xi_max_cqt, sigma_max_cqt)
s_max_noncqt, _ = s_w_from_xi_sigma(xi_min_noncqt, sigma_min_noncqt)

sci_widths = np.logspace(np.log10(s_min_cqt), np.log10(s_max_noncqt), n_psis)
sci_wav = lambda *a, **k: scipy.signal.morlet2(*a, **k, w=w_cqt)
lib_args['scipy'] = dict(data=x, wavelet=sci_wav, widths=sci_widths)

#%% PyWavelets ###############################################################
# pywt's wavelet generation makes it impossible to replicate arbitrary
# filterbanks, and approximation is difficult to decode, so we build something
# close enough, unfortunately forcing hard-coding here, to `Q, J = 16, 12`.
# Note, this designs for integrated wavelets, but CWT's output is then
# differentiated; the difference is small and negligible for our purposes.
pywt_wav = 'cmor10.0-2.7'
pywt_scales = np.logspace(np.log10(5.5), np.log10(5.5*fmax_fmin_ratio), n_psis)
lib_args['PyWavelets'] = dict(data=x, wavelet=pywt_wav, scales=pywt_scales,
                              method='fft')

#%% Gather & bench ###########################################################
# helper
def cwt_with_pad_and_unpad(cwt_fn, args):
    # need to pad and unpad manually if the library doesn't provide it
    x = args['data']
    pad_right = (N_psi - len(x)) // 2
    pad_left = N_psi - pad_right - len(x)
    args['data'] = np.pad(x, [pad_left, pad_right])

    out = cwt_fn(**args)
    if isinstance(out, tuple):
        out = out[0]
    out = out[..., pad_left:-pad_right]
    # restore original to dict
    args['data'] = x
    return out

# cwt & bench functions
def cwt_wavespin(**args):
    _ = sc.cwt(**args)

def cwt_ssqueezepy(**args):
    out = ssq.cwt(**args)[0]
    out = out[..., ::hop_size]

def cwt_scipy(**args):
    out = cwt_with_pad_and_unpad(scipy.signal.cwt, args)
    out = out[..., ::hop_size]

def cwt_pywavelets(**args):
    out = cwt_with_pad_and_unpad(pywt.cwt, args)
    out = out[..., ::hop_size]

cwt_fns = {
    'WaveSpin': cwt_wavespin,
    'ssqueezepy': cwt_ssqueezepy,
    'PyWavelets': cwt_pywavelets,
    'scipy': cwt_scipy,
}
bench_fns = {name: lambda name=name: cwt_fns[name](**lib_args[name])
             for name in cwt_fns}

#%%###########################################################################
# Run benchmarks & visualize
# --------------------------
title = f"CWT: len(x)={N}, hop_size={hop_size}"

results = run_benchmarks(bench_fns, n_iters, verbose=True)
viz_benchmarks(results, title)
