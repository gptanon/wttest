# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Synthetic signal generation."""
import numpy as np
import scipy.signal
from ...utils.measures import compute_bw_idxs


def echirp(N, fmin=1, fmax=None, tmin=0, tmax=1):
    """https://overlordgolddragon.github.io/test-signals/ (bottom)"""
    fmax = fmax or N // 2
    t = np.linspace(tmin, tmax, N)

    phi = _echirp_fn(fmin, fmax, tmin, tmax)(t)
    return np.cos(phi)


def _echirp_fn(fmin, fmax, tmin=0, tmax=1):
    a = (fmin**tmax / fmax**tmin) ** (1/(tmax - tmin))
    b = fmax**(1/tmax) * (1/a)**(1/tmax)
    phi_fn = lambda t: 2*np.pi * (a/np.log(b)) * (b**t - b**tmin)
    return phi_fn


def fdts(N, n_partials=2, total_shift=None, f0=None, seg_len=None,
         partials_f_sep=1.6, global_shift=0, brick_spectrum=False,
         endpoint=False):
    """Generate windowed tones with Frequency-dependent Time Shifts (FDTS)."""
    def brick(g):
        gf = np.fft.rfft(g)

        # center at dc
        ct = np.argmax(np.abs(gf))
        gf_ct = np.roll(gf, -ct)
        agf_ct = np.abs(gf_ct)
        # brickwall width = ~support width
        # decays slower so pick smaller criterion_amplitude
        width = np.where(agf_ct < agf_ct.max() / 10000)[0][0]
        brick_f = np.zeros(len(g)//2 + 1)
        brick_f[:width] = 1
        brick_f[-width:] = 1
        gf_ct *= brick_f

        gf_bricked = np.roll(gf_ct, ct)
        g_bricked = np.fft.irfft(gf_bricked)
        return g_bricked

    total_shift = total_shift or N//16
    f0 = f0 or N//12
    seg_len = seg_len or N//8

    t = np.linspace(0, 1, N, endpoint=endpoint)
    window = scipy.signal.tukey(seg_len, alpha=0.5)
    pad_right = (N - len(window)) // 2
    pad_left = N - len(window) - pad_right
    window = np.pad(window, [pad_left, pad_right])

    x = np.zeros(N)
    xs = x.copy()
    for p in range(0, n_partials):
        f_shift = partials_f_sep**p
        x_partial = np.sin(2*np.pi * f0 * f_shift * t) * window
        if brick_spectrum:
            x_partial = brick(x_partial)

        partial_shift = int(total_shift * np.log2(f_shift) / np.log2(n_partials))
        xs_partial = np.roll(x_partial, partial_shift)
        x += x_partial
        xs += xs_partial

    if global_shift:
        x = np.roll(x, global_shift)
        xs = np.roll(xs, global_shift)
    return x, xs


def bag_o_waves(N, names=None, sc=None, e_th=.02):
    """Collection of various signals useful for edge case testing.

    Parameters
    ----------
    N : int
        Length of signals to generate.

    names : list[str] / None
        Names of signals to generate. "Supported `names`" below.

    sc : Scattering1D / TimeFrequencyScattering1D
        Required for any `names` containing 'adversarial'.

    e_th : float
        Parameter for signals whose names contain 'adversarial'.

    Returns
    -------
    waves : dict[str: tensor/list[tensor]]
        Name-signal pairs.

    Supported `names`
    -----------------

      - `'randn'`: White Gaussian Noise, 0-mean, 1-std.
          General-purpose "if it works on this, it works".

      - `'pink'`: pink noise, i.e. WGN whose spectrum is scaled by 1/freq^2.
          Much more "uniform" in spectral energy distribution for scattering,
          since CWT frequencies are log-scaled. That is, same energy per unit
          frequency interval doesn't mean same energy per unit log-frequency:
          it'll be much less for lower frequencies, inflating the effective SNR.

      - `'impulse'`: unit impulse, centered. `x=zeros(N); x[N//2] = 1`.
          Possibly the chief "adversarial example" for scattering total energy
          conservation. Reliable surrogate for "silences" in data, which can lose
          surprising amounts of energy.
          Must be paired with `pad_mode='zero'` for intended results.

      - `'constant'`: flat line. `x=ones(N)`.
          Is both the best and the worst case for energy conservation.
          All of energy will be in zeroth order, so perfect conservation, but
          zeroth order is typically tossed out.
          Must be paired with `pad_mode='reflect'` for intended results.

      - `'impulse-train'`: uniform sequence of impulses. `x=zeros(N); x[::spc]=1`.
          Requires many second-order coefficients if first order has high
          time resolution (low `Q` / high `r_psi`) to not lose information.
          Good edge case for high time localization.

      - `'adversarial-am'`: amplitude-modulated cosines tailored to `sc`.
          Good (but not optimal) adversarial example for second-order energy
          conservation, used for determining whether second-order wavelets can
          be safely excluded.

          See `help(adversarial_am)`. This is multiple signals in a list.

      - `'adversarial-impulse-train'`: uniformly-spaced deltas tailored to `sc`.
          Is the one-signal version of 'adversarial-am'.
    """
    supported = {'randn', 'pink', 'impulse', 'constant', 'impulse-train',
                 'adversarial-am', 'adversarial-impulse-train'}
    if names is None:  # no-cov
        if sc is not None:
            names = supported
        else:
            names = [nm for nm in supported if 'adversarial' not in nm]
    elif not isinstance(names, (tuple, list)):
        assert isinstance(names, str), type(names)
        names = [names]
    if (any(name.startswith('adversarial') for name in names) and
            sc is None):  # no-cov
        raise ValueError("adversarial examples require `sc`. Got `names`\n%s" %
                         names)

    waves = {}
    for name in names:
        if name == 'randn':
            x = np.random.randn(N)
        elif name == 'pink':
            x = colored_noise(N, 'pink')
        elif name == 'impulse':
            x = np.zeros(N)
            x[N//2] = 1
        elif name == 'constant':
            x = np.ones(N)
        elif name == 'impulse-train':
            x = np.zeros(N)
            interval = max(N//64, 2)
            x[::interval] = 1
        elif name == 'adversarial-am' and sc is not None:
            x = adversarial_am(sc, e_th=e_th, N=N)
        elif name == 'adversarial-impulse-train' and sc is not None:
            x = adversarial_impulse_train(sc, e_th=e_th, N=N)
        waves[name] = x
    return waves


def adversarial_am(sc, e_th=.02, N=None):
    """Amplitude-modulated cosines that maximize the energy of high-frequency
    envelopes resulting from convolving with high-frequency wavelets. Useful as
    simplified worst-case for testing whether second-order wavelets can be safely
    excluded, based on the resulting loss of energy. It is far from worst case
    obtained by gradient descent, but is also much better than WGN.

    Note, to target energy loss as fraction of *total* scattering energy, `e_th`
    should be increased (e.g. x2), as `e_th` itself targets the fraction lost at
    second order, and greater `e_th` increases second-order's energy for this
    signal.

    Parameters
    ----------
    sc : Scattering1D / TimeFrequencyScattering1D
        Scattering instance.

    e_th : float
        Energy threshold. Will place AM bins at points where each first-order
        wavelet's energy drops below this (or, more precisely, the ratio of
        energy outside this interval to energy inside is `e_th`).

        Note, since worst case can't have multiple bins for same AM, we create
        multiple signals such that their energy supports don't overlap.

    N : int / None
        Intended length of input signal, if different from `sc.N`
        (note, cannot exceed max length of wavelets, `len(sc.psi1_f[0][0])`).

    Returns
    -------
    x_all : list[tensor]
        List of 1D arrays.
    """
    # arg check
    psi1_len = len(sc.psi1_f[0][0])
    if psi1_len < N:  # no-cov
        raise ValueError("Wavelets shorter than input! {} < {}".format(
            psi1_len, N))

    # set reference via highest freq wavelet
    R = lambda r: r / (1 - r)
    bw_idx_left0 = compute_bw_idxs(sc.psi1_f[0][0],
                                   criterion_amplitude=np.sqrt(R(e_th/2)))[0]

    # find number of distinct signals required according to the number of wavelets
    # it takes for bandwidths to no longer overlap per `e_th`
    for n1, psi1 in enumerate(sc.psi1_f):
        if n1 == 0:
            continue
        bw_idx_right = compute_bw_idxs(psi1[0],
                                       criterion_amplitude=np.sqrt(R(e_th/2)))[1]
        if bw_idx_right < bw_idx_left0:
            break
    n_signals = n1 + 1

    # adjust peak index by lengths so the padded signal peaks where intended
    len_adj = int(np.ceil(psi1_len / N))
    # sampling vector; `=True` enables reflect-pad to create perfect sine
    t = np.linspace(0, 1, N, endpoint=True)

    # generate AMs
    x_all = [np.zeros(N) for _ in range(n_signals)]
    for n1, psi1 in enumerate(sc.psi1_f):
        p1 = psi1[0]

        # fetch analytic-only bw and peak idx
        i0, i1 = compute_bw_idxs(p1, criterion_amplitude=np.sqrt(R(e_th)))
        i1 = min(i1, len(p1)//2)
        bw1 = i1 - i0 + 1  # add 1 in case the "defender" is conservative
        peak1 = psi1['peak_idx'][0] // len_adj
        # need half bw as it doubles about carrier after modulation;
        # also can't have the result spill over Nyquist
        half_bw1 = min(max(bw1 // 2, 1), N//2 - peak1)

        a = np.cos(2*np.pi * half_bw1 * t)
        c = np.cos(2*np.pi * peak1    * t)
        x_all[n1 % n_signals] += a * c

    return x_all


def adversarial_impulse_train(sc, e_th=.02, N=None):
    """Uniformly spaced Dirac-deltas that maximize the energy of high-frequency
    envelopes resulting from convolving with high-frequency wavelets. Useful as
    worst-case for testing whether second-order wavelets can be safely excluded,
    based on the resulting loss of energy.

    This is the one-signal alternative of `adversarial_am` that targets the
    maximum bandwidth, since highest frequencies are the likeliest to be
    excluded in various applications. Also see "Note" in `adversarial_am`.

    Parameters
    ----------
    sc : Scattering1D / TimeFrequencyScattering1D
        Scattering instance.

    e_th : float
        Energy threshold. Will place impulse bins in freq domain
        (impulse train in time <=> impulse train in freq)
        at this threshold for the highest bandwidth wavelet.

        The threshold is such that the ratio of energy outside the resulting
        frequency interval to energy inside of it `e_th`.

    N : int / None
        Intended length of input signal, if different from `sc.N`
        (note, cannot exceed max length of wavelets, `len(sc.psi1_f[0][0])`).

    Returns
    -------
    x : tensor
        Impulse train.
    """
    if N is None:  # no-cov
        N = sc.N

    # fetch bandwidths, only the analytic side
    R = lambda r: r / (1 - r)
    bw1_all = []
    for p1 in sc.psi1_f:
        i0, i1 = compute_bw_idxs(p1[0], criterion_amplitude=np.sqrt(R(e_th)))
        i1 = min(i1 + 1, len(p1[0])//2)  # +1 in case of conservative "defender"
        bw1_all.append(i1 - i0)
    # take max
    bw_max = max(bw1_all)
    # temporal spacing required to achieve the bandwidth
    tm_interval = N // bw_max
    # take next power of 2; enables perfect impulse train via reflect-pad.
    # "next" since targeting higher bw may yield all pure sines in scalogram
    tm_interval = int(2**np.ceil(np.log2(tm_interval)))
    # adjust for padding; here take N's closest power of 2 since we're trying
    # to maximize match with padded FFT
    pad_adj = 2**sc.J_pad // int(2**np.round(np.log2(N)))
    tm_interval *= pad_adj

    # make train
    x = np.zeros(N)
    x[::tm_interval] = 1

    return x


def colored_noise(N, color='pink'):
    """Pink or brown noise."""
    exponent = {'pink': 1., 'brown': 2.}[color]

    # gaussian with phase
    xf = np.random.randn(N//2+1) + 1j*np.random.randn(N//2+1)
    xf[:1].imag = 0  # imag dc zero for real `x`

    # scale spectrum
    scl = np.linspace(0, .5, N//2 + 1)
    scl[0] = scl[1]  # reuse bin1 for dc
    scl = 1/scl**(exponent / 2)
    xf *= scl

    # time, norm
    x = np.fft.irfft(xf, n=N)
    x /= x.std()
    return x
