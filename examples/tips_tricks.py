# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
Tips and Tricks
===============

    1. Helpful library utilities demos
    2. Python tips for non-veterans
"""

###############################################################################
# Setup
# -----
import numpy as np
import wavespin
from wavespin import Scattering1D
from wavespin.visuals import plot, plotscat

sc = Scattering1D(2048)

#%%############################################################################
# WaveSpin tips
# -------------
# Time scattering and JTFS both have `DYNAMIC_PARAMETERS` and
# `REACTIVE_PARAMETERS`. We can do `help(sc)` then `Ctrl + F` and look them up,
# but basically, "dynamics" can be changed after `sc` was created, and "reactives"
# likewise but it must be done through `sc.update()`. Let's see what they are:
print(sc.DYNAMIC_PARAMETERS)   # for JTFS it's `DYNAMIC_PARAMETERS_JTFS`
print(sc.REACTIVE_PARAMETERS)  # again, and for all following all-caps examples

#%%############################################################################
# As of writing, `out_type` is the only one which can be updated directly:
sc.out_type = 'array'

#%%############################################################################
# Updating others with same syntax will error. Instead, we do:
sc.update(oversampling=1, pad_mode='zero')

#%%############################################################################
# We can also fetch the hidden keyword arguments as
print(sc.SUPPORTED_KWARGS)

#%%############################################################################
# and their defaults as
print(sc.DEFAULT_KWARGS)

#%%############################################################################
# There's in fact a third level of control, if you know what you're doing:
from pprint import pprint  # pretty-print

pprint(wavespin.CFG)
wavespin.CFG['S1D']['sigma0'] = 0.1  # docs in `help(wavespin.configs)`
# let's re-set in this case
wavespin.configs.restore_defaults()

#%%############################################################################
# For appropriate backends, we can move between CPU and GPU as `sc.gpu()` and
# `sc.cpu()`, or move to a specific device via `sc.to_device(device_spec)`.
# We can move back and forth without a problem.
# Non-NumPy tensors can be easily converted to NumPy with the following:
import torch
from wavespin.utils.gen_utils import npy

xt = torch.tensor([5., 2.])
if torch.cuda.is_available():
    xt = xt.cuda()
xn = npy(xt)
print(xn, xn.dtype)

#%%############################################################################
# WaveSpin quick facts
# --------------------
#   - NumPy backend defaults to double precision, all else to single. There's no
#     particular reason for this, `precision='single'` works just fine.
#   - All backends support GPU and differentiability, except NumPy.
#   - Tuple batch shape is supported for inputs, e.g. `(10, 2, 8192)` if we
#     have 10 audio files each with 2 channels. Though not all utilities support
#     this (or batching at all), which is worked around by flattening or looping.
#   - `Scattering1D` default `r_psi` isn't great for `cwt`; should be `>0.8`.
#     Check effects on resolution via `sc.info()`.
#   - CWT and time scattering are part-CQT, part-STFT (approx).
#     See "CQT" in "Terminology" in `help(wavespin.Scattering1D())`.
#   - "warnings" show only once per configuration, so their absence doesn't mean
#     everything's fine. https://stackoverflow.com/q/30489449/10133797
#   - Docs say `help(wavespin.Scattering1D())` for shorthand, but at least one
#     argument is needed; you could use `help(wavespin.Scattering1D(99))`.
#   - Feature performance (classification etc), speed, and memory are all
#     subject to drastic improvements... if the library's developed further!

#%%############################################################################
# Python tips
# -----------
# First and foremost, use an IDE! I recommend Spyder, here's a quick tutorial.
# TODO
#
# Everything is an object in Python. Python comes with nice built-ins for
# lobotomizing them; let's make a dummy:
#
class Dog():
    ears = 2

    def __init__(self):
        self.name = "Chewy"

    def bark(self):
        print("woof")

d = Dog()
print(vars(d))

#%%############################################################################
# `vars(obj)` fetches (instance) attributes as a dictionary. `dir(obj)` is more
# complete, with methods and attributes:
print(dir(d))

#%%############################################################################
# but it only fetches the names. We can also print values, while skipping
# built-in magic methods", via:
for name in dir(d):
    if '__' not in name:
        print(name, '--', getattr(d, name))

#%%############################################################################
# Applied to scattering, we can inspect the `meta` object without reading the
# documentation:
meta = sc.meta()
print(type(meta))

#%%############################################################################
# It's a dict, a collection of key-value pairs: `dict[key] == value`. `list(obj)`
# iterates an iterable object - dict is keys-first, which lets us inspect the
# options:
print(list(meta))
print(type(meta['xi']))

#%%############################################################################
# So, `meta` seems to be a dict of arrays. Get its shape, and example value:
print(meta['xi'].shape)
print(meta['xi'][0])

#%%############################################################################
# A `nan`, wonder why? Check docs!
help(sc.meta)

#%%############################################################################
# Ok, it says to check a specific file, so let's do that:
help(wavespin.scattering1d.scat_utils.compute_meta_scattering)

#%%############################################################################
# So, `nan` means no valid value, which makes sense, as zeroth order has no
# second-order meta.
#
# Let's apply our tools to `sc` itself, and limit the printed size of dict values:
for key, value in vars(sc).items():
    print(key, '--', str(value)[:50])

#%%############################################################################
# Seems our filters are stored at `psi1_f`, `psi2_f`, and `phi_f`. Let's look
# at `psi2_f`:
print(type(sc.psi2_f))
print(len(sc.psi2_f))

#%%############################################################################
# A list of what appears to be several filters. Inspect one randomly:
print(type(sc.psi2_f[3]))
print(list(sc.psi2_f[3]))

#%%############################################################################
# So each entry is a dict with what appears to be filter meta, except the
# integers. Let's check `0` and `1`:
print(type(sc.psi2_f[5][0]))
print(type(sc.psi2_f[5][1]))
print(sc.psi2_f[5][0].shape)
print(sc.psi2_f[5][1].shape)

#%%############################################################################
# They're arrays, probably our filters, and of different lengths. Indeed the
# integers are subsampling factors, which we'd learn from docs, but let's move on
# and plot a few:
plot(sc.psi2_f[1][0])
plot(sc.psi2_f[2][0], title="Second-order filters", color='tab:blue', show=True)

#%%############################################################################
# Let's plot one in time, and center it:
p_t = np.fft.ifftshift(np.fft.ifft(sc.psi1_f[-1][0]))
plot(p_t, title="Largest first-order filter, in time", complex=2, show=True)

#%%############################################################################
# Now let's fetch all first-order frequencies with a "list comprehension", and
# plot!
xi1s = [p['xi'] for p in sc.psi1_f]
plotscat(xi1s, title="First-order center frequencies", show=True)

#%%############################################################################
# Lastly, the following is very handy for repeated calls - "iterable unpacking":
def func(a, b, c=0, d=0, e=0, f=0, g=0):
    print(a, b, c, d, e, f)

common_args = (1, 2, 3)
common_kwargs = dict(e=5, f=6)

func(*common_args, d=4, **common_kwargs)
# only `*` ordering matters
func(*common_args, **common_kwargs, d=4, g=-1)
