# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""
Applied Design
==============
We configure time scattering and JTFS to suit a real-world signal in a
domain-agnostic manner.
"""

###############################################################################
# How to read & Purpose
# ---------------------
# This example is meant to be stepped through and understood. It's not meant
# to be a cheat sheet, but can be used as such (though docs do it better).
#
# The intent is to show how to think about core parameters, and how to use library
# tools to assist in inspecting a given configuration and its effect on a signal.

###############################################################################
# Setup
# -----
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, ifft, ifftshift
import scipy.signal

import wavespin
from wavespin import Scattering1D, TimeFrequencyScattering1D
from wavespin.visuals import plot, plotscat, scalogram, imshow, viz_jtfs_2d

#%%############################################################################
# Load trumpet and inspect it with CWT
# ------------------------------------
# Load trumpet, duration 2.9 seconds (sampling rate, sr=22050)
# generated via `librosa.load(librosa.ex('trumpet'))[0][:int(2.9*22050)]`
x = np.load('librosa_trumpet.npy')
# to reduce run time of this example, decimate; not much is lost in this case
decimation_factor = 2
x = scipy.signal.decimate(x, decimation_factor, ftype='fir')
fs = 22050 / decimation_factor
N = x.shape[-1]

# We can use any configuration for starters, but we set a few params here
# to keep the example forward-compatible
precision = 'single'
J = int(np.ceil(np.log2(N))) - 3
ckw = dict(shape=N, precision=precision)  # common keyword arguments
sc0 = Scattering1D(Q=8, J=J, **ckw)

scalogram(x, sc0, fs, show_x=True)
# can also do `Wx = sc.cwt(x); imshow(Wx, abs=1)`

###############################################################################
# We observe:
#
#   1. Vanishing energy below ~360 Hz
#   2. The longest rise or fall in frequency lasts about 1 second, but useful
#      time-frequency geometries span potentially the entire signal
#   3. The time-frequency resolution is satisfactory, but we should check others

#%%############################################################################
# Tune `Q`
# --------
# Try higher frequency resolution
sc1 = Scattering1D(Q=16, J=J, **ckw)
scalogram(x, sc1, fs)

#%%############################################################################
# Indeed this is better, the different ridges ("curves") are better separated.
# Let's try another step up:
sc2 = Scattering1D(Q=24, J=J, **ckw)
scalogram(x, sc2, fs)

###############################################################################
# It's yet another improvement, but upon closer look, we start to see loss of
# time separation creeping in, and there's enough separation along frequency,
# so we should probably stop here. Choose `Q=24`, and clear the previous objects
# to save memory.
del sc0, sc1, sc2

#%%############################################################################
# Tune `J`
# --------
# Per observation 1, many wavelets are wasted in tiling the frequency axis
# according to CQT (see example "Parameter Sweeps"). Lowering `J` will reduce
# the number of lower-frequency wavelets, but also make the largest wavelet
# smaller, which is relevant for observation 2. Let's experiment:
Q = 24
ckw['Q'] = Q
sc3 = Scattering1D(J=7, **ckw)
scalogram(x, sc3, fs)

#%%############################################################################
# Much better! Note, the lower frequencies aren't lost, they're just sparsely
# tiled; the plot won't show it. We could keep loweing `J`, but let's stop here
# and inspect the largest wavelet. We could print or plot - let's do both;
# recall, higher index = lower freq.
p = sc3.psi1_f[-1]
print("support = {:.3g} sec".format(p['support'][0] / fs))
pt = ifftshift(ifft(p[0]))

duration = len(pt) / fs
t = np.linspace(-duration/2, duration/2, len(pt), endpoint=False)
plot(t, pt, complex=2, show=True)

#%%############################################################################
# Terrible! The longest wavelet doesn't even last half a second. Large-scale
# convolutions are among the most important advantages of CWT over STFT and
# conv-nets; a large CNN kernel is many weights, slow, and prone to overfitting,
# while a wavelet is well-behaved and fast via FFT convolution - we should take
# advantage whenever possible.
#
# Note, the `'support'` meta suggests a much longer duration; that's because
# of the long tail induced by the `analytic=True` configuration - it's an
# accurate measure for how it's defined, but not useful to us. A more robust
# metric for wavelet's concentration is `'width'`, which translates to support
# as approximately multiplied by 8 (for Gabors):
print("support = {:.3g} sec (width-based)".format(p['width'][0] * 8 / fs))

#%%############################################################################
# Here lies trouble: simply increasing the largest scale (`J`) won't help as
# the large scale wavelet's frequency is much lower than the signal's lowest
# frequency, so nothing's captured. The alternative is to enlarge all wavelets
# (increase `Q`), but we saw that has downsides.
#
# To rescue, depth: increase `J2`. While there's no suitable large-scale spectral
# content right in the signal, it is in signal's AM/FM envelopes, i.e. second
# order. Even without higher `J2`, higher orders increase the receptive field
# of the overall network, exactly like in conv-nets.
#
# Increasing `J` by `1` doubles the width of the largest wavelet, so increasing
# by `4` is x16 and should put us where we want.

# `average=False` to reuse `sc4` later; that needs `'list'`
J = (7, 11)
ckw['J'] = J
sc4 = Scattering1D(average=False, out_type='list', **ckw)

#%%############################################################################
# We'll see no effect in the scalogram since it's computed with first-order
# wavelets, so instead show the largest scale wavelet from second order;
# as a bonus, show the support of the input signal:
p = sc4.psi2_f[-1]
pt = ifftshift(ifft(p[0]))

# note, only matches input's duration for visual, not start and end
input_start = t[len(t)//2 - len(x)//2]
input_end   = t[len(t)//2 + len(x)//2]
plot(t, pt, complex=2, vlines=([input_start, input_end], {'linewidth': 2}),
     show=True)
# for wavelets affected by `analytic=True`, this will undershoot, but not by a lot
print("support = {:.3g} sec (width-based)".format(p['width'][0] * 8 / fs))

###############################################################################
# Much better. For time scattering, it means extracting AM/FM structures >1 sec
# in duration. For JTFS, it's same, but enhanced with the joint 2D time-frequency
# geometry, and assigning lower FM slopes to longer durations (a useful prior).

#%%############################################################################
# Inspect second-order time
# -------------------------
# Do unaveraged to inspect detailed second-order geometry. Need to disable
# subsampling to enable concatenation.

sc4.update(oversampling=99)                # make all subsampling factors 0
Scx = sc4(x)                               # scatter
Scx = np.array([c['coef'] for c in Scx])   # concatenate into array
Scx2 = Scx[sc4.meta()['order'] == 2]       # fetch second order

imshow(Scx2, abs=1,
       title="S2 (unrolled)",
       xlabel="time [index]",
       ylabel="(n2, n1) [index]")

#%%############################################################################
# This is an "unrolled" representation, where all second-order coefficients
# are flattened along the same axis.
#
# Let's also look at it on per-second order wavelet basis; as per energy flow,
# more first-order rows are discarded for higher frequencies in second-order -
# the number that's kept we mark as `n_xi1s`.

def viz_scat1d_2d(sc, Scx, fs):
    Scx2 = Scx[sc.meta()['order'] == 2]
    # fetch generating wavelet indices
    ns = sc.meta()['n']
    # unique second-order
    n2s = np.unique([int(n[0]) for n in ns if not np.isnan(n[0])])
    # number of first-order done per second-order
    n_n1s_for_n2 = {n2: sum(ns[:, 0] == n2) for n2 in n2s}

    # make canvas
    fig, axes = plt.subplots(len(n2s), 1, sharex=True, sharey=False,
                             figsize=(12, 24))
    # keep colormap scaling same across subplots
    vmax = Scx2.max() * .9
    # reusable
    label_kw = dict(weight='bold', fontsize=14)

    # plot for each `n2`
    start = 0
    for i, n2 in enumerate(n2s):
        ax = axes[i]
        p2 = sc.psi2_f[n2]
        end = start + n_n1s_for_n2[n2]
        ax.imshow(Scx2[start:end], cmap='turbo', aspect='auto', vmin=0, vmax=vmax)
        start = end

        # styling
        title = "xi2={:.3g} Hz | n_xi1s={}".format(
            p2['xi'] * fs, n_n1s_for_n2[n2])
        ax.set_title(title, loc='left', **label_kw, y=.985)
        n1s = ns[ns[:, 0] == n2].astype(int)[:, 1]
        xi1_max = "%.3g" % (sc.psi1_f[n1s[0] ]['xi'] * fs)
        xi1_min = "%.3g" % (sc.psi1_f[n1s[-1]]['xi'] * fs)
        ax.set_yticks([0, len(n1s) - 1])
        ax.set_yticklabels([xi1_max, xi1_min])
        ax.set_ylabel("xi1 [Hz]", **label_kw)
        ax.yaxis.set_label_coords(-.02, .5)

    # style x-axis
    n_xlabels = 10
    t_sig = ["%.3g" % tk for tk in
             np.linspace(0, len(x)/fs, len(x))[::len(x)//n_xlabels]]
    x_ticks = np.arange(len(x))[::len(x)//n_xlabels]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(t_sig)
    ax.set_xlabel("time [sec]", **label_kw)

    # remove gaps
    fig.subplots_adjust(hspace=0.22)
    plt.show()

viz_scat1d_2d(sc4, Scx, fs)

##############################################################################
# We observe a decent amount of ridge variation (curvy stuff), which a subsequent
# learned spatial operator could explot (e.g. CNNs) if we don't collapse the
# time axis (global average). The low `xi2` coefficients are high in energy, so
# we're successfully utilizing large-scale convolutions.

#%%############################################################################
# Tune `T`
# --------
# This is arguably the most important parameter to tune. Ideally, we account for
# both the domain and task (e.g. audio, acoustic scene recognition is more
# permissive than discriminating vowels), and subsequent processing steps (e.g.
# CNN makes much better use of more time coefficients than linear classifier).
#
# There are also important domain-agnostic considerations:
#
#   - Higher `T` works better with higher `Q` and `J`. That's because larger
#     scale implies larger temporal envelope, and lower bandwidth, meaning lower
#     average frequency of A.M. envelopes generated at first order, and thuswise
#     in subsequent orders. That's a global shift in energies toward lower orders,
#     meaning less information is lost for a given amount of averaging. It's also
#     related to why `Q2 > 1` is a bad idea for large `Q1`.
#   - `T >= 2**J` is required for time-warp stability of the whole representation.
#     If `J` is close to `log2(N)`, this is less important, as generally warps
#     are less of an issue for very low frequencies (or more pertinently, large
#     scales).
#
# A good ballpark for "fair" amount of leftover time coefficients is 8 to 16.
# That means `N / T` is 8 or 16, or e.g. `T = N / 16`, so `T = 1998`. Unless
# we have a specific reason to be precise, we should round that to a power of 2,
# since the library subsamples only by (previous) powers of 2, which would yield
# an oversampled representation. So, `T=4096`. Conveniently, that makes
# `log2(T) = 12`, which matches `J2=12` - but that's just a coincidence. We
# could've simply chosen `T=2**J2`, or based on desired output size (as we did).
T = 2048
ckw['T'] = T

del sc4

#%%############################################################################
# Now JTFS
# --------
# We'll get back to `Scattering1D`, but feel free to skip ahead. Let's inspect
# JTFS with the more information-rich variant, `out_3D=True`, and carry over our
# configs up to this point (we don't have to, but nearly everything said so far
# also applies to JTFS). `out_3D` requires `average_fr=True`, but as before,
# let's look at the pre-averaged representation; `F=1` will be visually
# indistinguishable. We should do this with both `equalize_pairs=False`, to get
# a sense of true relative energies, and `True`, to better see resulting
# geometries.

# `viz_jtfs_2d` requires `'dict:' in out_type`
# use `Q_fr=1` for speed
jtfs = TimeFrequencyScattering1D(**ckw, out_3D=True, average_fr=True, F=1,
                                 Q_fr=1, out_type='dict:array')
Scx = jtfs(x)
viz_jtfs_2d(jtfs, Scx, fs=fs, viz_filterbank=False, equalize_pairs=False)

##############################################################################
# Substantial amount of spinned energy, doing good.

#%%
viz_jtfs_2d(jtfs, Scx, fs=fs, viz_filterbank=False, equalize_pairs=True)

#%%############################################################################
# Substantial amount of 2D detail that can be exploited by a 2D conv-net, and
# enough spin assymetry (difference in energies between up and down) for good
# performance under frequential averaging.
#
# Here there's no pressing need, but if some pairs' energies don't behave
# as we expect, we can inspect some numbers.

# Remind ourselves what visuals are available
print(dir(wavespin.visuals))
# Plot
_ = wavespin.visuals.energy_profile_jtfs(Scx, jtfs.meta(), x)

###############################################################################
# Those familiar with JTFS energy analysis may find the result odd. If curious
# or unfamiliar, check the docs! Basically, we should check back once we have
# `F` and `J_fr` that we'll actually use.

#%%############################################################################
# Tuning `F`
# ----------
# As noted in docs, setting `F` and `J_fr` relative to the frequential equivalent
# of `N` (`N_frs_max)` requires a dummy instantiation (see docs for how that's
# done). We note the following:
#
#    - Like with `T`, `F` should take into account domain and task requirements.
#      The units of `F * Q1` are `cycles / octave`, so there's a physical
#      dependence (see units note in `help(wavespin.visuals.viz_jtfs_2d)`).
#    - `N_frs_max` is closely proportional to `Q1`.
#    - Like with `T`, `F` can be set based on desired output size (though
#      this requires knowledge of `N_frs_max`).
#    - Low `F`, like `4`, is nearly lossess on large joint slices, for reasons
#      we won't cover. Though it also depends on `Q1` and `r_psi1`. Losses with
#      larger `F` can be contained with `sampling_phi_fr='recalibrate'` and
#      `F_kind='decimate'`.
#    - `F='global'` always works.
#
# Here we simply roll with `16` to balance output size against loss of
# information, and sticking with the rule of thumb of `8` or `16` output units
# (at time of writing, `N_frs_max==98`).
F = 8

#%%############################################################################
# Tuning `J_fr`
# -------------
# We follow `J2`'s logic, along frequency. Note, `Q1`-dependence and some other
# points about `F` also apply here.
#
# For this, let's show the scalogram with the frequency axis in samples instead
# of physical units:
Wx = jtfs.cwt(x)
imshow(Wx, abs=1)

###############################################################################
# As of writing, `len(Wx) == 98`, and the longest vertically spanning structure
# appears to span 80 rows. 128 is larger than the scalogram, but we don't really
# mind the tails of wavelet support, and it's more important to get high energy
# over the structures of interest. Recalling our earlier calculation, that makes
# the target width `128 / 8 = 16`, or `J_fr = 4`. And friendly reminder, the
# padded scalogram is what's fed to compute joint pairs!
J_fr = 4

#%%############################################################################
# Tuning `Q_fr`
# -------------
# Advanced topic, but in short, if compute burden isn't a concern, `Q_fr=2`
# should work better in general, especially with higher-dim convs or high `F`.
# We take `Q_fr=1` for speed.
Q_fr = 1

#%%############################################################################
# Tuning `smart_paths`, `paths_exclude`
# -------------------------------------
# Our visuals suggest there's pretty much zero energy in high `xi2` paths;
# we can gain speed and memory by dropping them. A simple approach is using
# `paths_exclude['n2']`, but we can't just say `n2=0` as that's already
# excluded; we can check what's included either via meta, or
print(list(jtfs.paths_include_n2n1))

###############################################################################
# Ok, so the right-most `n2` in our plot is the lowest `n2`, which is
# (as of writing) `n2=4`, so `pe = jtfs.paths_exclude; pe['n2'].append(4)`
# and then `jtfs.update(paths_exclude=pe)` (see `help(jtfs.update)`).
#
# Instead, it's better to tune `smart_paths`. We can keep cranking it until
# `n2=4` disappears - here we just roll with `0.02`, which happens to work.
smart_paths = .02

#%%############################################################################
# Putting it all together
# -----------------------
# Let's finalize our network and inspect it toe-to-toe, reproducing all visuals
# in one place for convenience:
jtfs = TimeFrequencyScattering1D(**ckw, F=F, J_fr=J_fr, Q_fr=Q_fr,
                                 smart_paths=smart_paths, out_3D=True,
                                 average_fr=True, out_type='dict:array')
Scx = jtfs(x)

#%%############################################################################
# Visualize filters
# -----------------
# Scalogram isn't a "filter" but it's what's fed to joint scattering, so it's
# quite relevant in "that which determines output":
scalogram(x, jtfs, show_x=False, fs=fs)

#%%############################################################################

# Temporal filterbank --------------------------------------------------------
wavespin.visuals.filterbank_scattering(jtfs, second_order=True, lp_sum=True)
wavespin.visuals.filterbank_heatmap(jtfs, first_order=True, second_order=True,
                                    parts=['abs'], w=.9)

# Frequential filterbank -----------------------------------------------------
wavespin.visuals.filterbank_jtfs_1d(jtfs, lp_sum=True, zoom=-1)
wavespin.visuals.filterbank_heatmap(jtfs, frequential=True, parts=['abs'], w=.9)

#%%############################################################################

# Joint filterbank -----------------------------------------------------------
viz_jtfs_2d(jtfs, fs=fs)

#%%############################################################################

# Largest support wavelets ---------------------------------------------------
# fetch time-domain
pt1 = ifftshift(ifft(jtfs.psi1_f[-1][0]))
pt2 = ifftshift(ifft(jtfs.psi2_f[-1][0]))
ptf = ifftshift(ifft(jtfs.psi1_f_fr_up[0][-1].squeeze()))
# frequential equivalent of `input_start`, `input_end`
i0 = len(ptf)//2 - jtfs.scf.N_frs_max//2
i1 = len(ptf)//2 + jtfs.scf.N_frs_max//2 + 1

# plot -----------------------------------------------------------------------
# pack common args
pkw = dict(complex=2, vlines=([input_start, input_end], {'linewidth': 2}),
           show=True)
# plot
plot(t, pt1, title="Largest wavelet, first-order", **pkw)

#%%############################################################################
plot(t, pt2, title="Largest wavelet, second-order", **pkw)

#%%############################################################################
plot(ptf, title="Largest wavelet, frequential scattering",
     vlines=([i0, i1], {'linewidth': 2}), complex=2, show=True)

#%%############################################################################
# Visualize outputs
# -----------------

# Zeroth order (omit units)
plotscat(Scx['S0'].squeeze(), title="Zeroth order", show=True)

# First order (omit units)
imshow(Scx['S1'].squeeze(), title="First order", abs=1, show=True)

# Joint
viz_jtfs_2d(jtfs, Scx, fs=fs, viz_filterbank=False, equalize_pairs=False)
viz_jtfs_2d(jtfs, Scx, fs=fs, viz_filterbank=False, equalize_pairs=True)
_ = wavespin.visuals.energy_profile_jtfs(Scx, jtfs.meta(), x)

###############################################################################
# At this point one can try another signal or configuration. We won't explain
# every new plot here, but there's docs and code for the curious. For this
# signal, we've achieved a really good design; all that's missing is suitable
# output normalization, not covered here (see e.g. JTFS 2D conv-net example).

#%%############################################################################
# Dataset handling
# ----------------
# **But I've got more than one signal!** Right.
#
# This requires some devised numeric measures we can apply to a given signal
# in automated fashion, that can be iterated over multiple signals and then
# summarized for final viewing. Advanced topic, but we provide a template
# and one such measure here.
#
# Let's track the mean, min, and max of absolute value of real-FFT, to get a
# sense of spectral energy contents. Recall, CWT is just multiplications in
# frequency domain, so if FFT is zeros over a frequency interval, so is CWT
# (hence scattering). Refer to code comments on methodology specifics.
#

class StatsFFT():
    """Tracks mean, min, and max of |rDFT|^2."""
    def __init__(self, N, dataset_size):
        self.dataset_size = dataset_size

        fft_size = N // 2 + 1
        # the only valid instantiation
        self.fmean = np.zeros(fft_size)
        # play safe and instantiate to something large
        self.fmin = np.ones(fft_size) * 1e9
        # play safe and instantiate to lowest possible value
        self.fmax = np.zeros(fft_size)

    def __call__(self, data):
        # `|rDFT|`
        axf = np.abs(rfft(data))**2
        # normalize energy to unity to make it independent of input scaling
        # and focus on spectral shape
        axf /= np.linalg.norm(axf)

        # it's more performant to divide once later, but safer to do it each
        # time to avoid numeric overflow if there's lots of `x`
        self.fmean += axf / self.dataset_size
        self.fmin = np.min([self.fmin, axf], axis=0)
        self.fmax = np.max([self.fmax, axf], axis=0)

###############################################################################
# Now make our data generator to loop over. Here we keep it simple and just
# reuse our trumpet, alongside white Gaussian noise. More likely, we'll load
# from a directory; for this, see "Generator example" in
# `help(wavespin.toolkit.fit_smart_paths)`.
np.random.seed(0)
x_all = [x, np.random.randn(len(x))]
sf = StatsFFT(len(x), len(x_all))

for data in x_all:
    sf(data)

#%%############################################################################
# Now, visualize! As our transform is log-scaled, we're very much interested
# in the log version, so plot it alongside linear:
def plot_along_logscaled(x, title):
    from wavespin.modules._visuals.primitives import _gscale
    fig, axes = plt.subplots(1, 2, sharey=True)
    plot(x, title=title, ax=axes[0], fig=fig)
    plot(x, title="log-scaled", logx=1, ax=axes[1], fig=fig)

    fig.subplots_adjust(wspace=.05)
    fig.set_size_inches((12*_gscale(), 5*_gscale()))
    plt.show()

plot_along_logscaled(sf.fmean, "|rFFT|^2 dataset mean")
#%%############################################################################
plot_along_logscaled(sf.fmin,  "|rFFT|^2 dataset minimum")
#%%############################################################################
plot_along_logscaled(sf.fmax,  "|rFFT|^2 dataset maximum")

###############################################################################
# We treat this as a fully-fledged dataset and not just two examples. The mean
# plots undermine our former design choice to restrict `J1`, as there's clearly
# significant energy in lower frequencies in examples we didn't inspect. As per
# former analysis, that's not sufficient to increase `J1`, our filterbank already
# tiles low frequencies - we should check that there's large-scale structure worth
# extracting at first order. Though, it's very unlikely that the answer is "no",
# and even if it's "yes", a larger `J1` is still worth it for other reasons -
# check docs. It seldom harms to just use `J[0] = log2(N) - 3`.
#
# Importantly, a great way to tune `smart_paths` is via
# `wavespin.toolkit.fit_smart_paths` - check its docs.

#%%############################################################################
# Wrapping up
# -----------
# If you've read this, both intros, and the Stack Exchange articles - congrats!
# You've graduated "JohnM's Scattering School". Your cranium has expanded.
#
# Respective parameter docs advise further, worth checking.
#