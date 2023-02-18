# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Filter measures: support, width, etc, in 1D.

All methods are general and work for any 1D filter/waveform/sequence,
except `compute_spatial_width`.
"""
import math
import warnings
import numpy as np
from scipy.fft import fft, ifft, ifftshift
from .algos import smallest_interval_over_threshold_indices
from .gen_utils import is_real


def compute_spatial_support(pf, criterion_amplitude=1e-3, guarantee_decay=False):
    """Compute spatial support of `pf` as the interval, in number of samples,
    where sum of envelope (absolute value) inside it is `1 / criterion_amplitude`
    times greater than outside.

    Used for avoiding boundary effects and incomplete filter decay. Accounts for
    tail decay, but with lessser weight than the Heisgenberg resolution measure
    for monotonic decays (i.e. no bumps/spikes). For prioritizing the main lobe,
    see `wavespin.utils.measures.compute_temporal_width()`.

    Parameters
    ----------
    pf : np.ndarray, 1D
        Filter, in frequency domain.

        Assumes that the time-domain waveform, `ifft(pf)`, is centered at index
        `0` (for Morlets & Gaussians, this means `argmax(abs(ifft(pf))) == 0`).

    criterion_amplitude : float
        Computes `criterion_ratio` as `1/criterion_amplitude`.

    guarantee_decay : bool (default False)
        In practice, this is identical to `True`, though `True` will still return
        a slightly more conservative estimate for Morlets & Gaussians.
        See `help(wavespin.utils.measures._integral_ratio_bound)`.

    Returns
    -------
    support : int
        Total temporal support of `pf`, measured in samples ("total" as opposed to
        only right or left part).

    Note
    ----
    The simple version of this algorithm is to find the first point, away from
    center of `pt` (ifft(pf)), that drops below `criterion_amplitude * pt.max()`,
    and multiply by 2. That yields the same result as `guarantee_decay=True` for
    Morlets & Gaussians.

    Details
    -------
    Mathematically, this bounds the point-wise (at each shift) error of
    convolution in terms of relative contributions of each point (each dot-mult)
    to the aggregation step. In continuous time, it's an L1 integral

        $A = \int_{start}^{end} |\psi(t)| dt$
        $B = \int_{-\infty}^{\infty} |\psi(t)| dt$

    with `start` and `end` such that `A / (B - A) > 1 / criterion_amplitude`.
    That is, for `criterion_amplitude=1e-3`, it's `inside / outside > 1000`.
    Hence, all points outside the support interval, combined, contribute x1000
    less to the result than all the points inside. `support == end - start`.

    Alternative formulation
    -----------------------

    Find `S = support` such that

        A = ||conv(x, \psi_S) - conv(x, \psi)||
        B = ||conv(x, \psi)||
        A / B < criterion_amplitude

    that is, the relative Euclidean distance between the infinite duration
    (completely untrimmed) filter and the filter trimmed to S, is less than
    `criterion_amplitude`. This is well-motivated, and practically the first
    formulation warrants the second, with the spec of `criterion_amplitude=1e-3`
    in former yielding `1e-3` to `1e-2` in latter. However, it is ultimately
    ill-conditioned and lacks utility:

        1. Any change in `psi` can be exploited to maximize the total convolution
           Euclidean distance well beyond the intended `criterion_amplitude`.
           This is confirmed with gradient descent.
           Practically, this isn't a major point, but this incompleteness limits
           precise filter design and additional criteria are required.

        2. The criterion can be satisfied even in cases of incomplete decay,
           yielding finite `support < N`. This defeats the purpose.

        3. This measure offers no direct description on how much padding should
           be used to avoid boundary effects, as it is an aggregate measure over
           the entire convolution, whereas boundary effects are localized.

        4. The result of interest unpads, hence includes contributions of padding,
           and yields different bounds depending on `pad_mode`. An upper bound can
           be found via padless circular convolution (surrogate for any general
           padding), but for `S > len(x_unpadded)` the discrepancy is great.
    """
    assert pf.ndim == 1 or (pf.ndim == 2 and pf.shape[-1] == 1), pf.shape
    if pf.ndim == 2:
        pf = pf.squeeze(-1)
    # center about N//2 to compute "support" over a contiguous interval
    pt = ifftshift(ifft(pf))

    support = _integral_ratio_bound(
        pt, criterion_ratio=1/criterion_amplitude, measure='abs',
        center_input=False, c='peak', return_sided=False,
        guarantee_decay=guarantee_decay)
    return support


def compute_minimum_required_length(fn, N_init, max_N=None,
                                    criterion_amplitude=1e-3):
    """Computes minimum required number of samples for `fn(N)` to have spatial
    support less than `N`, as determined by `compute_spatial_support`.

    Parameters
    ----------
    fn: FunctionType
        Function / lambda taking `N` as input and returning a 1D filter in
        frequency domain.

    N_init: int
        Initial input to `fn`, will keep doubling until `N == max_N` or
        temporal support of `fn` is `< N`.

    max_N: int / None
        See `N_init`; if None, will raise `N` indefinitely.

    criterion_amplitude : float
        value \\epsilon controlling the numerical
        error. The larger criterion_amplitude, the smaller the temporal
        support and the larger the numerical error. Defaults to 1e-3

    Returns
    -------
    N: int
        Minimum required number of samples for `fn(N)` to have temporal
        support less than `N`.
    """
    N = 2**math.ceil(math.log2(N_init))  # ensure pow 2
    while True:
        try:
            pf = fn(N)
        except ValueError as e:  # get_normalizing_factor()
            if "division" not in str(e):  # no-cov
                raise e
            N *= 2
            continue

        # must guarantee decay else we risk failing "constant line test" and
        # returning N - 1, N - 2, etc, yet we threshold `pf_support < N`.
        pf_support = compute_spatial_support(
            pf, criterion_amplitude=criterion_amplitude, guarantee_decay=True)

        if N > 1e9:  # no-cov
            # avoid crash
            raise Exception("couldn't satisfy stop criterion before `N > 1e9`; "
                            "check `fn`")
        if pf_support < N or (max_N is not None and N > max_N):
            break
        N *= 2
    return N


def compute_filter_redundancy(p0_f, p1_f):
    """Measures "redundancy" as normalized overlap of energies. Namely, ratio of
    product of energies to mean of energies of frequency-domain filters
    `p0_f` and `p1_f`.
    """
    p0sq, p1sq = np.abs(p0_f)**2, np.abs(p1_f)**2
    # equalize peaks
    p0sq /= p0sq.max()
    p1sq /= p1sq.max()

    # energy overlap relative to sum of individual energies
    r = np.sum(p0sq * p1sq) / ((p0sq.sum() + p1sq.sum()) / 2)
    return r


def r_psi_to_redundancy(r_psi, upper_half_only=True):
    """Converts `r_psi` to what's computed by `compute_filter_redundancy`.
    Determined by interpolating tested filters; very strong fit, MSE<6.2e-6.

    upper_half_only:
        - True is for r_psi 0.5 to 0.99, False 0.0001 to 0.99.
        - Default is `True` as that's the better fit over that interval,
          and we don't expect user `r_psi` to drop below `0.5`
          (if it does, there's warnings).

    Excludes the `Q=1` case, as there excess bandwidth distorts Morlet's bell, for
    either `analytic`. For `r_psi=sqrt(.5)`, we get `0.1633` instead of `0.1767`.
    """
    return ((0.9152*r_psi + 0.001244)**4 if upper_half_only else
            (0.9163*r_psi + 0.0004136)**4)


def compute_spatial_width(p_f, N=None, pts_per_scale=6, fast=None,
                          sigma0=.13, criterion_amplitude=1e-3):
    """Measures "width" in terms of amount of invariance imposed via convolution.
    Prioritizes spatial main lobe over tails. See below for detailed description.

    Parameters
    ----------
    p_f : np.ndarray, 1D
        Frequency-domain filter of length >= N. "Length" must be along dim0,
        i.e. `(freqs, ...)`.

    N : int / None
        Unpadded output length (unstrided). (In scattering we convolve at e.g.
        x4 input's length, then unpad to match input's length).
        Defaults to `len(p_f) // 2`, corresponding to padding `x` to next power
        of 2.

    pts_per_scale : int
        Used only in `fast=False`: number of Gaussians generated per dyadic
        scale. Greater improves approximation accuracy but is slower.

    fast : bool
        Fast approximation of the full algorithm. See "Fast algorithm".

        Defaults to `True` if `sigma0` and `criterion_amplitude` match
        pre-computed values.

    sigma0, criterion_amplitude : float, float
        See `help(wavespin.scattering1d.filter_bank.gauss_1d)`. Parameters
        defining the Gaussian lowpass used as reference for computing `width`.
        That is, `width` is defined *in terms of* this Gaussian.

    Returns
    -------
    width: int[>=1]
        The estimated width of `p_f`, in number of samples.

    Motivation
    ----------
    The measure is, a relative L2 distance (Euclidean distance relative to
    input norms) between inner products of `p_f` with an input, at different
    time shifts (i.e. `L2(A, B); A = sum(p(t) * x(t)), B = sum(p(t) * x(t - T))`).

      - The time shift is made to be "maximal", i.e. half of unpadded output
        length (`N/2`), which provides the most sensitive measure to "width".
      - The input is a Dirac delta, thus we're really measuring distances between
        impulse responses of `p_f`, or of `p_f` with itself. This provides a
        measure that's close to that of input being WGN, many-trials averaged.

    This yields `l2_reference`. It's then compared against `l2_Ts`, which is
    a list of measures obtained the same way except replacing `p_f` with a fully
    decayed (undistorted) Gaussian lowpass - and the `l2_T` which most closely
    matches `l2_reference` is taken to be the `width`.

    The interpretation is, "width" of `p_f` is the width of the (properly decayed)
    Gaussian lowpass that imposes the same amount of invariance that `p_f` does.

    Algorithm
    ---------
    The actual algorithm is different, since L2 is a problematic measure with
    unpadding; there's ambiguity: `T=1` can be measured as "closer" to `T=64`
    than `T=63`). Namely, we take inner product (similarity) of `p_f` with
    Gaussians at varied `T`, and the largest such product is the `width`.
    The result is very close to that described under "Motivation", but without
    the ambiguity that requires correction steps.

    Fast algorithm
    --------------
    Approximates `fast=False`. If `p_f` is fully decayed, the result is exactly
    same as `fast=False`. If `p_f` is very wide (nearly global average), the
    result is also same as `fast=False`. The disagreement is in the intermediate
    zone, but is not significant.

    We compute `ratio = p_t.max() / p_t.min()`, and compare against a fully
    decayed reference. For the intermediate zone, a quadratic interpolation
    is used to approximate `fast=False`.

    Assumption
    ----------
    `abs(p_t)`, where `p_t = ifft(p_f)`, is assumed to be Gaussian-like.
    An ideal measure is devoid of such an assumption, but is difficult to devise
    in finite sample settings.

    Note
    ----
    `width` saturates at `N` past a certain point in "incomplete decay" regime.
    The intution is, in incomplete decay, the measure of width is ambiguous:
    for one, we lack compact support. For other, the case `width == N` is a
    global averaging (simple mean, dc, etc), which is really `width = inf`.

    If we let the algorithm run with unrestricted `T_max`, we'll see `width`
    estimates blow up as `T -> N` - but we only seek to quantify invariance
    up to `N`. Also, Gaussians of widths `T = N - const` and `T = N` have
    very close L2 measures up to a generous `const`; see `test_global_averaging()`
    in `tests/scattering1d/test_jtfs.py` for `T = N - 1` (and try e.g. `N - 64`).
    """
    from ..scattering1d.filter_bank import gauss_1d

    # edge case
    if len(p_f) == 1:  # no-cov
        return 1

    # obtain temporal filter
    p_f = p_f.squeeze()
    p_t = np.abs(ifft(p_f))

    # relevant params
    Np = len(p_f)
    if N is None:
        N = Np // 2
    ca = dict(criterion_amplitude=criterion_amplitude)

    # compute "complete decay" factor
    uses_defaults = bool(sigma0 in (.1, .13) and criterion_amplitude == 1e-3)
    if fast is None:
        fast = bool(uses_defaults)
        if not fast:
            warnings.warn("Significant initialization slowdown expected per "
                          "non-default `sigma0` or `criterion_amplitude`.")

    if uses_defaults:
        # precomputed; keyed by `sigma0`
        # to add values, simply execute `else` for a given pair of below params
        complete_decay_factor = {.1: 16,
                                 .13: 16}[sigma0]
        fast_approx_amp_ratio = {.1:  0.8208687174155399,
                                 .13: 0.7163455413697654}[sigma0]
    else:
        if fast:  # no-cov
            raise ValueError("`fast` requires using default values of "
                             "`sigma0` and `criterion_amplitude`.")
        # compute phi at `T` that equates `p_f`'s length (max permitted `T`).
        # the resulting `complete_decay_factor` specifies at what length
        # `gauss_1d` should be sampled, relative to `p_f`, to attain complete
        # decay in the *worst case* (max `T`).
        # lastly, `fast_approx_amp_ratio` is the worst case decay ratio - that is,
        # if we exceed this, it means we've not fully decayed for our given length
        # (`T` of `p_f` is too great relative to `len(p_f)`). if we don't exceed
        # this, then we can compute the `T` easily directly from `p_f`
        # (see `else` of `if r_min > fast_approx_amp_ratio`)
        T = Np
        phi_f_fn = lambda Np_phi: gauss_1d(Np_phi, sigma0 / T)
        Np_min = compute_minimum_required_length(phi_f_fn, Np, **ca)
        complete_decay_factor = 2 ** math.ceil(math.log2(Np_min / Np))
        phi_t = np.abs(ifft(phi_f_fn(Np_min)))
        if fast:
            fast_approx_amp_ratio = phi_t[T] / phi_t[0]

    if fast:
        ratio = (p_t / p_t[0])[:len(p_t)//2]  # assume ~symmetry about halflength
        rmin = ratio.min()
        if rmin > fast_approx_amp_ratio:
            # equivalent of `not complete_decay`

            th_global_avg = .96
            # never sufficiently decays
            if rmin > th_global_avg:
                return N
            # quadratic interpolation
            # y0 = A + B*x0^2
            # y1 = A + B*x1^2
            # B = (y0 - y1) / (x0^2 - x1^2)
            # A = y0 - B*x0^2
            y0 = .5 * Np
            y1 = N
            x0 = fast_approx_amp_ratio
            x1 = th_global_avg
            B = (y0 - y1) / (x0**2 - x1**2)
            A = y0 - B*x0**2
            T_est = A + B*rmin**2
            # do not exceed `N`
            width = int(round(min(T_est, N)))
        else:
            width = np.argmin(np.abs(ratio - fast_approx_amp_ratio))
        width = max(width, 1)
        return width

    # if complete decay, search within length's scale
    support = compute_spatial_support(p_f, **ca, guarantee_decay=True)
    complete_decay = bool(support != Np)
    too_short = bool(N == 2 or Np == 2)
    if too_short:
        return (1 if complete_decay else 2)
    elif not complete_decay:
        # cannot exceed by definition
        T_max = N
        # if it were less, we'd have `complete_decay`
        T_min = 2 ** math.ceil(math.log2(Np / complete_decay_factor))
    else:  # complete decay
        # if it were more, we'd have `not complete_decay`
        # the `+ 1` is to be safe in edge cases
        T_max = 2 ** math.ceil(math.log2(Np / complete_decay_factor) + 1)
        # follows from relation of `complete_decay_factor` to `width`
        # `width \propto support`, `support = complete_decay_factor * stuff`
        # (asm. full decay); so approx `support ~= complete_decay_factor * width`
        T_min = 2 ** math.floor(math.log2(support / complete_decay_factor))
    T_min = max(min(T_min, T_max // 2), 1)  # ensure max > min and T_min >= 1
    T_max = max(T_max, 2)  # ensure T_max >= 2
    T_min_orig, T_max_orig = T_min, T_max

    n_scales = math.log2(T_max) - math.log2(T_min)
    search_pts = int(n_scales * pts_per_scale)

    # search T ###############################################################
    def search_T(T_min, T_max, search_pts, log):
        Ts = (np.linspace(T_min, T_max, search_pts) if not log else
              np.logspace(np.log10(T_min), np.log10(T_max), search_pts))
        Ts = np.unique(np.round(Ts).astype(int))

        Ts_done = []
        corrs = []
        for T_test in Ts:
            N_phi = max(int(T_test * complete_decay_factor), Np)
            # not tested with single so use double precision
            phi_f = gauss_1d(N_phi, sigma=sigma0 / T_test, precision='double')
            phi_t = ifft(phi_f).real

            trim = min(min(len(p_t), len(phi_t))//2, N)
            p0, p1 = p_t[:trim], phi_t[:trim]
            p0 /= np.linalg.norm(p0)  # /= sqrt(sum(x**2))
            p1 /= np.linalg.norm(p1)
            corrs.append((p0 * p1).sum())
            Ts_done.append(T_test)

        T_est = int(round(Ts_done[np.argmax(corrs)]))
        T_stride = int(Ts[1] - Ts[0])
        return T_est, T_stride

    # first search in log space
    T_est, _ = search_T(T_min, T_max, search_pts, log=True)
    # refine search, now in linear space
    T_min = max(2**math.floor(math.log2(max(T_est - 1, 1))), T_min_orig)
    # +1 to ensure T_min != T_max
    T_max = min(2**math.ceil(math.log2(T_est + 1)), T_max_orig)
    # only one scale now
    search_pts = pts_per_scale
    T_est, T_stride = search_T(T_min, T_max, search_pts, log=False)
    # only within one zoom
    diff = pts_per_scale // 2
    T_min, T_max = max(T_est - diff, 1), max(T_est + diff - 1, 3)
    T_est, _ = search_T(T_min, T_max, search_pts, log=False)

    width = max(T_est, 1)
    return width


def compute_bandwidth(pf, criterion_amplitude=1e-3, return_sided=False,
                      center_input=True, c='peak', guarantee_decay=False):
    """Compute bandwidth of `pf` as the interval, in number of samples, where
    energy inside of it is `(1/criterion_amplitude)**2` times greater than
    outside.

    Computes the "information bandwidth", measuring the greatest possible
    subsampling factor without losing information (within tolerance). It holds
    irrespective of center frequency (there may be aliasing, but
    aliasing != lossy). With a sufficient sampling rate, this directly corresponds
    to bandwidth of the unaliased and (approx.) bandlimited original continuous
    frequency-domain function.

    Parameters
    ----------
    pf : np.ndarray, 1D
        Filter, in frequency domain.

    criterion_amplitude : float
        Computes `criterion_ratio` as `(1/criterion_amplitude)**2`.
        The `1/` is just for code clarity, what matters is the `**2`.

    return_sided : bool (default False)
        Whether to parts of bandwidth to left and right of `c` separately.
        The parts will sum to one greater than the single result returned by
        `False`, as they each include `c`.

    center_input : bool (default True)
        Will center `pf` about `c`, that is, `np.roll(pf, -c + len(pf)//2)`.

        The motivation is to disambiguate "bandwidth". For `pf` that's any
        general waveform, which can be aliased or non-bandlimited, what counts
        as "bandwidth" isn't uniquely-defined. In continuous-time, a pre-aliased,
        bandlimited `pf` lies on a *contiguous* interval between sampling bounds
        (Nyquist zones). Centering `pf` hence allows the computation of largest
        such contiguous interval, where circularity is deliberately disregarded.

        An example of ambiguity is a Morlet near DC, which typically leaks into
        negatives. Without centering, this yields maximum bandwidth, `== N - 1`,
        as the spectrum spans all the way from bin 1 (no DC) to -1. Yet, this
        result isn't useful in any important sense, and contradicts the basic
        notion that high bandwidth = high time localization.

    c : str['peak'] / int
        Specifies the centering index, since it's not necessarily the peak
        in the general case. `'peak'` sets it to `argmax(abs(pf))`.

    guarantee_decay : bool (default False)
        In practice, this is identical to `True`, though `True` returns a
        slightly more conservative estimate for Morlets & Gaussians.
        See `help(wavespin.utils.measures._integral_ratio_bound)`.

    Returns
    -------
    bandwidth : int
        Bandwidth of `pf`, measured in samples.

    Details
    -------
    The measure is identical to that of `compute_spatial_support` except it
    uses L2^2 (energy) rather than L1.

    Energy because it's the conserved quantity. That is, it's the only quantity
    that's conserved between the spatial and frequency domains, via
    Parseval-Plancherel's theorem. Hence a measure in one domain can be used
    as a measure of both - in this case, effect of aliasing on time-domain result.

    The measure quantifies aliasing in terms of the total effect of folding.
    The added (aliased) energies in total are, with `criterion_amplitude=1e-3`,
    x1,000,000 less than energies being added onto. If the filter is

        [p0, p1, p2, p3, p4, p5, p6, p7]

    then subsampling by 2 makes it

        [p0 + p4, p1 + p5, p2 + p6, p3 + p7]

    and we seek to keep this result as close as possible to

        [p0, p1, p6, p7]

    which we accomplish via

        A = sum(abs([p0, p1, p6, p7])**2)
        B = sum(abs([p2, p3, p4, p5])**2)
        A / B > (1 / criterion_amplitude)**2

    Alternative formulation
    -----------------------
    For aliasing, ideally we'd guarantee that the ratio of aliased and unaliased
    energies is controlled by `criterion_amplitude`, but that's not what this
    measure does, as bin energies aren't linear in addition:
    `(p0 + p1)**2 != p0**2 + p1**2`. However, it's close enough in practice,
    and this alt metric has important drawbacks:

        1. It doesn't interpret as a generalized information measure, as it
           depends on center frequency in addition to bandwidth.
           E.g. two Gaussians with identical `sigma` but different frequency
           shifts will have different bandwidths.
        2. It isn't suited for measuring losses due to lowpass filtering
           (note, this doesn't involve subsampling or aliasing), whereas the used
           metric (energy) can directly quantify it in time-domain losses.
    """
    out = _integral_ratio_bound(
        pf, criterion_ratio=1/criterion_amplitude, measure='energy',
        center_input=center_input, return_sided=return_sided,
        guarantee_decay=guarantee_decay, c=c)

    if return_sided:
        bw_left, bw_right = out
        assert bw_left >= 0 and bw_right >= 0, (bw_left, bw_right)
        assert bw_left > 0 or bw_right > 0, (bw_left, bw_right)
        return bw_left, bw_right
    else:
        # == 0 is possible but we'll never need it (it's `zeros()`)
        bandwidth = out
        assert bandwidth > 0, bandwidth
        return bandwidth


def compute_bw_idxs(pf, criterion_amplitude=1e-3, c='peak'):
    """Computes the start and end index of the frequential support of `pf`,
    as defined by `wavespin.utils.measures.compute_bandwidth()`.

    Parameters
    ----------
    See `wavespin.utils.measures.compute_bandwidth()`.

    Returns
    -------
    bw_idxs : tuple[int]
        `idx_left, idx_right = bw_idxs`. The support is sliced inclusively,
        i.e. `pf[idx_left:idx_right + 1]`. If `idx_left < 0`, then slicing
        is circular: `hstack([pf[idx_left:], pf[:idx_right + 1])`.

    Indexing formulation
    --------------------
    `bw_idxs` slices `pf` contiguously, including circularly. This introduces
    negative indexing. The conversion is, where `l, r = bw_idxs` and `r` is
    never negative:

    ::
        if l < 0:
            slc = np.hstack([pf[l:], pf[:r + 1]])
        else:
            slc = pf[l:r + 1]

    Indexing non-uniqueness
    -----------------------
    "Bandwidth" here follows the definition of `compute_bandwidth`. The indexing
    hence isn't uniquely-defined for any arbitrary 1D sequence `pf`, even with `c`
    known, since determination of `l` and `r` is non-unique.

        - An example is a ramp, `np.arange(16)`, which is what's approximately
          produced by a heavily aliased Morlet, e.g. `morlet_1d(256, .99, .99)`.
          The most natural indexing is `pf[:N]`, yet if we follow the evolution
          of the Morlet, this always changes, starting from the unaliased case
          which wraps from right to left. In this example it's unclear what is
          "correct" to begin with, but it's to illustrate the problem.
    """
    # arg check
    pf = np.abs(pf)
    peak_idx = np.argmax(pf)
    N = len(pf)

    # handle analytic/anti-
    anti_analytic = bool(peak_idx > N//2)
    if anti_analytic:
        pf[1:] = pf[1:][::-1]  # freq reversal, make into analytic
        peak_idx = np.argmax(pf)

    if c == 'peak':
        c = peak_idx
    elif anti_analytic and c != 0:
        c = N - c

    # main computation #######################################################
    # compute bw from each side of peak, then indices w.r.t. peak.
    bw_left, bw_right = compute_bandwidth(
        pf=pf, criterion_amplitude=criterion_amplitude, center_input=True,
        guarantee_decay=False, return_sided=True, c=c)

    # this undoes `b_left =` and `b_right =` in `_integral_ratio_bound`,
    # with `idx_left <=> start` and `idx_right <=> end`.
    # `cic` is what's used in `_integral_ratio_bound`, where we
    # `roll(, -c + N//2)`, so to restore original indices, undo this shift
    cic = N//2
    start = (cic - bw_left + 1) + (c - cic)
    end   = (bw_right + cic) + (c - cic)
    idx_left, idx_right = start, end

    if bw_left <= c + 1:
        # if `bw_left` exceeds `c + 1`, it means after shifting to center we've
        # wrapped around from negatives, then it's ok to start at negatives
        assert idx_left >= 0, idx_left
    if c + bw_right > N and idx_left >= 0:
        # if `c + bw_right` exceeds input length, it means after shifting to
        # center we've wrapped around from positives. The (c - cic) correction
        # above won't acknowledge the wrapping, so we do it here
        idx_left = -(N - idx_left)
        idx_right = idx_right - N
    assert idx_right <= N, idx_right  # can't exceed length

    # adjust so that it's the last point *within* the support,
    # i.e. the support is to be sliced inclusively, `slice(left, right + 1)`
    idx_right -= 1

    # map back to anti-analytic ##############################################
    if anti_analytic:
        l, r = idx_left, idx_right
        lzero = bool(l == 0)  # edge case, zeor not flippable
        # make each positive
        l = N - l if l < 0 else l
        r = N - r if r < 0 else r

        # since we're flipping excluding index 0
        M = N - 1
        # map to M-indexed space
        l, r = l - 1, r - 1
        # swap left and right, M-1 converts distance to index
        r, l = (M - 1) - l, (M - 1) - r
        # revert M's indices to N's indices
        r, l = r + 1, l + 1

        # e.g. (-3, 7), so slicing from far right end onto left
        if r < 0:
            # the analytic flip pf[1:][::-1] and rest of steps can't yield
            # `idx_right < 0`, here meaning `l < 0`
            assert l >= 0, (l, r, idx_left, idx_right, start, end, c, N)
            l, r = -l, -r
            l = -(N + l)
        if r == N and lzero:
            # to make sense of the adjustment, increment `idx_left` from negatives
            # until zero
            l, r = l - N, 0
        idx_left, idx_right = l, r

        # sanity checks
        info = (idx_left, idx_right, c, N)
        assert 0 <= idx_right < N,     info
        assert 0 <= abs(idx_left) < N, info
        assert idx_right >= idx_left,  info
        # all of the above is a long-winded way of doing
        #     l, r = -r, -l
        #     if r < 0: l, r = N + l, N + r
        # but with reasoning presented

    return idx_left, idx_right


def compute_max_dyadic_subsampling(pf, bw_idxs, real=None, analytic=None,
                                   for_scattering=True):
    """Computes maximum permitted dyadic subsampling upon `pf` ~without aliasing,
    via bandwidth indices that depend on `criterion_amplitude`, which controls
    alias error.

    To subsample length `N` by 2 without aliasing, we require having frequencies
    no greater than `N//4 - 1`. With analytic filters, this is relaxed to `N//4`.
    Relevant is `test_compute_max_dyadic_subsampling_and_fft_upsample` in
    `tests/scattering1d/test_measures.py`.

    This function is based on Kymatio's `get_max_dyadic_subsampling`, and
    for scattering, is wrong. It works well enough for `Q >= 8`. It will be
    corrected in a future release.

    Parameters
    ----------
    pf : tensor
        1D filter in frequency domain.

    bw_idxs : tuple[int]
        Output of `wavespin.utils.measures.compute_bw_idxs()`.

    real : bool / None
        Whether `pf` is real-valued in time domain, i.e. `ifft(pf)`.
        Makes small adjustment on subsampling criterion, together with `analytic`
        and `for_scattering`.

        Will determine automatically if `None`, but it's recommended to specify
        this as the determination may be compute-expensive.

    analytic : bool / None
        Whether `pf` is strictly analytic or anti-analytic.
        Makes small adjustment on subsampling criterion, together with `real`
        and `for_scattering`.

        Will determine automatically if `None`, but it's recommended to specify
        this as the determination may be compute-expensive.

    for_scattering : bool (default True)
        Makes small adjustment on subsampling criterion, together with `real`
        and `analytic`. See "Subsampling criterion: general vs analytic".

    Returns
    -------
    j : int
        Max dyadic subsampling factor.

    Subsampling criterion: general vs analytic
    ------------------------------------------
    Suppose N=8, and we subsample by 2. Since subsampling <=> folding, we have

        [p0, p1, p2, p3,
         p4, p5, p6, p7]
        -->
        [p0 + p4, p1 + p5, p2 + p6, p3 + p7]

    We require

        [p0, p1, p2, p3, p4, p5, p6, p7]
        -->
        [p0, p1, p6, p7]

    which is only achieved if p4=p5=p2=p3=0:

        [p0, p1, 0, 0, 0, 0, p6, p7]

    Yet, for `p` real in time, this requires `p2=p6`, which aliases. If `p`
    isn't real, then we can't tell whether third slot in subsampled came from
    `p6` or `p2`, or both. In short, having Nyquist bin == aliasing. Unless,
    that is, we *know* it came from `p2`, which we do due to analyticity.
    This does require `analytic=True`, but to not introduce additional dependence
    to scattering output size, we treat `analytic=False` the same via
    `for_scattering=True` - the difference is likely to be small anyway.

    The analytic case is thus

        [p0, p1, p2, 0, 0, 0, 0, 0] -->
        [p0, p1, p2, 0]

    Extended discussion: https://dsp.stackexchange.com/q/83781/50076

    Note: information bandwidth
    ---------------------------
    This measure strives to avoid aliasing, which is stricter than avoiding
    loss of information. Latter may vary with context depending on goal, but
    the most general measure provides the upper bound: N // bw.

    Here we take close to half of that upper bound. One can note, `bmax = bw - 1`
    in best case (bw can be yet larger), which holds with `analytic=True`, and
    we do `N // 2 / bmax`, so it's not exact halving. This concerns keeping
    positive and negative bins positive and negative, and the folding boundary
    is offest by DC and Nyquist bins, hence the adjustment.
    However, two special cases defy this pattern:

        - bw=1 -> N/2**j=1: no positives or negatives. Alpha is omega.
        - bw=2 -> N/2**j=2: positive is negative (Nyquist). Yin is yang.

    The `bw=2` case isn't coded explicitly due to a numeric coincidence:

        N // 2 / (bw - 1) == N // bw

    also since the code would be verbose.
    """
    j = _compute_max_dyadic_subsampling(pf, bw_idxs, real, analytic,
                                        for_scattering)
    # sanity check: cannot exceed information bandwidth
    # (bypassed by `for_scattering=True` as it permits aliasing)
    if not for_scattering:
        N = len(pf)
        bw = bw_idxs[1] - bw_idxs[0] + 1
        assert 2**j <= N // bw, (j, N, bw)

    return j


def _compute_max_dyadic_subsampling(pf, bw_idxs, real, analytic, for_scattering):
    # input checks ###########################################################
    N = len(pf)
    bi0, bi1 = bw_idxs
    # guaranteed by `compute_bw_idxs`
    assert 0 <= bi1 < N, bw_idxs       # [0, N)
    assert 0 <= abs(bi0) < N, bw_idxs  # [0, N)
    both_nonnegative = bool(bi0 >= 0 and bi1 >= 0)
    if both_nonnegative:
        assert bi1 >= bi0, bw_idxs     # [bi0, bi1]

    # special case handling ##################################################
    # excess bw
    if bi1 - bi0 > N // 2:
        return 0
    # slices Nyquist
    if both_nonnegative and bi0 <= N//2 and bi1 > N//2:
        return 0
    # DC-only
    if bi0 == bi1 == 0:
        return int(np.log2(N))

    # input handling (cont'd) ################################################
    # determine if real
    if real is None:
        # first try to compute quickly
        if -bi0 == bi1:  # necessary but not sufficient
            if np.allclose(pf[1], pf[-1]):  # necessary but not sufficient
                real = is_real(pf)
            else:
                real = False
        else:
            real = False

    # determine if analytic or anti-analytic
    a   = bool(both_nonnegative and bi0 <= bi1 <= N//2)
    aa0 = bool(both_nonnegative and bi1 >= bi0 >= N//2)
    aa1 = bool(bi0 < 0 and bi1 == 0)
    aa  = bool(aa0 or aa1)
    if analytic is None:
       analytic = bool(a or aa)

    # compute ################################################################
    # case: anti-analytic, convert to analytic equivalent
    if aa0:
        # should've been handled above
        assert bi0 >= N//2, (bi0, bi1)
        bi0, bi1 = N - bi0, N - bi1
    # case: negative inclusive index, same role as positive inclusive index
    if bi0 < 0:
        bi0 = -bi0

    # compute max permitted subsampling
    bmax = max(bi0, bi1)
    if real or (not analytic and not for_scattering):
        j = np.log2(N // 2 // (bmax + 1))
    else:
        j = np.log2(N // 2 // bmax)
    return int(j)


def _integral_ratio_bound(p, criterion_ratio=1e3, measure='abs',
                          center_input=False, c='peak', return_sided=False,
                          guarantee_decay=False):
    """Used for computing bandwidth or effective spatial support of a filter,
    or any general `p`.

    Integrates (sums) `p`, and finds where, in the interval around center of `p`,
    the ratio of contents inside the interval to outside is `> criterion_ratio`.

    Suppose `measure='abs'`, `N=len(p)`, and `ap = abs(p)`. Returns `end - start`
    such that

        `sum(ap[start:end]) / (sum(ap[:start]) + sum(ap[end:])) > criterion_ratio`

    Parameters
    ----------
    p : np.ndarray, 1D
        Filter, in target domain (frequency for bandwidth, time for
        boundary effects / decay for convolution).

    criterion_ratio : float
        Threshold for ratio on what to call "bound".

    measure : str['abs', 'energy']
        Will use

            - `'abs'`: `abs(p)`
            - `'energy'`: `abs(p)**2`, `criterion_ratio = criterion_ratio**2`

    center_input : bool (default False)
        Whether to center `p`, i.e. `np.roll(p, -c + len(p)//2)`.

    c : str['peak'] / int
        Specifies the centering index. `'peak'` sets it to `argmax(abs(p))`.

    return_sided : bool (default False)
        Return left and right parts of computation separately. Their sum
        is 1 greater than the result returned by `False`, per each including `c`.

    guarantee_decay : bool (default False)
        `True` additionally guarantees `interval` is such that no individual point
        outside of it exceeds `criterion_ratio` with respect to `p.max()`.
        Conversely, all points inside of `interval` do exceed.

        See "Decay note".

    Returns
    -------
    interval : int / tuple[int]
        Measured in number of samples.

    Decay note
    ----------
    With `guarantee_decay=False`, this method fails the "flat line test", in that
    we can pass in `np.ones(N)` and get `<N` for output, for a sufficiently large
    `N`. This is intentional, as the method assures that contents inside the
    yielded interval outweigh the contents outside of it by a factor of
    `criterion_ratio`. In context of

        - boundary effects / temporal decay, convolution: contents outside the
          interval will contribute at most `1/criterion_ratio` to the output.
        - aliasing, subsampling: frequencies outside the interval will overlap
          (fold, alias onto) with those inside the interval by at most
          `1/criterion_ratio**2` (measure='energy') of inside's energy.

    If this still isn't desired, an additional criterion guarantees "decay"
    occurs - that there is a point, `i`, that's below some fraction of the peak:

        p.max() / p[i] > criterion_ratio/2

    checked separately to left and to right of `c`. Ensures no point *after* `i`
    fails the ratio, i.e. it's really `p.max() / p[i:].max()`, to account for
    cases where `p` first decays and then rises again.

    Interval note
    -------------
    The method *defines* `interval` to span around index `len(p)//2`, so if we
    pass in something centered elsewhere with `center_input=False` or wrong `c`,
    the results are inaccurate.

    For example, `[1, 1, 0, 0, 0, 0, 0, 1]` outputs `8`, even though we may want
    `3`, but the `8` interpretation is valid for both convolution and aliasing.
    """
    # handle args ############################################################
    if p.size != 1:
        p = p.squeeze()
    N = len(p)
    # edge cases: not enough points to run algorithm
    if N == 1:
        return (0, 1) if return_sided else 1
    elif N == 2:
        # apply `guarantee_decay`'s criterion
        eps = np.finfo(p.dtype).eps * 100
        p = np.abs(p)
        idx0_nonzero = bool(p[0] / (p[1] + eps) < criterion_ratio)
        idx1_nonzero = bool(p[1] / (p[0] + eps) < criterion_ratio)
        if idx0_nonzero and idx1_nonzero:
            return (1, 2) if return_sided else 2
        else:  # don't acknowledge zero-input case
            return (1, 1) if return_sided else 1

    # ensure double precision to avoid numeric errors in `algos`
    p = p.astype(np.complex128 if 'complex' in p.dtype.name else
                 np.float64)
    # p -> abs(p), handle `c`
    p = np.abs(p)
    if c == 'peak':
        c = np.argmax(p)
    assert 0 <= c < N, (c, N)

    # center at N//2 (move `c` to N//2)
    if center_input:
        p = np.roll(p, -c + N//2)
        cic = N//2  # cic == "c_in_compute"
    else:
        cic = c

    # run algorithm ##########################################################
    # normalize for numeric stability
    pmax = p.max()
    if np.allclose(pmax, 0.):
        raise Exception("`p` is all zeros, can't compute.")
    p /= pmax
    if measure == 'energy':
        p = p**2

    # energy squares
    if measure == 'energy':
        criterion_ratio = criterion_ratio**2

    # below gives `r = interval / total`,
    # we want `R = interval / (total - interval)`.
    # They're related by `r = R / (R + 1)`. More precisely, it gives `r * total`.
    # The converse relation is `R = r / (1 - r)`.
    R = criterion_ratio
    r = R / (R + 1)
    # compute
    start, end = smallest_interval_over_threshold_indices(
        p, threshold=r * p.sum(), c=cic)

    # set maxima such that left and right intervals about `c` cannot spill
    # over the array
    b_left_max, b_right_max = cic + 1, N - cic
    if start == -1:
        # definition
        b_left = b_left_max
        b_right = b_right_max
    else:
        assert start <= cic < end, (start, cic, end)
        # include peak in both left and right halves, so they're standalone;
        # then total = b_left + b_right - 1.
        b_left = cic - start + 1
        b_right = end - cic

    # sanity checks
    assert 0 <= b_left  <= b_left_max,  (b_left,  b_left_max,  start, cic, end)
    assert 0 <= b_right <= b_right_max, (b_right, b_right_max, start, cic, end)

    if guarantee_decay:
        criterion_amplitude = 1 / (criterion_ratio / 2)
        # below should actually use `p / p.max()` but we've already normalized
        # `p.max()` to `1`, so save compute
        earliest_excess_left  = np.where(p[:cic]     > criterion_amplitude)[0]
        earliest_excess_right = np.where(p[cic + 1:] > criterion_amplitude)[0]

        # if there's no excess, all examined points qualify
        if len(earliest_excess_left) == 0:
            b_left_decayed = b_left
        else:
            earliest_excess_left = earliest_excess_left[0]
            # `+ 1` since `earliest_excess_*` are inclusive
            b_left_decayed = (cic - earliest_excess_left) + 1
        if len(earliest_excess_right) == 0:
            b_right_decayed = b_right
        else:
            earliest_excess_right = earliest_excess_right[-1] + cic + 1
            b_right_decayed = (earliest_excess_right - cic) + 1

        # `_decayed` shouldn't ever be less, but be sure
        b_left  = max(b_left,  b_left_decayed)
        b_right = max(b_right, b_right_decayed)

    # return #################################################################
    if return_sided:
        return b_left, b_right

    if b_left == b_left_max and b_right == b_right_max:
        interval = N  # definition
    else:
        # `-1` to not double-count the center
        interval = b_left + b_right - 1
    return interval


# short ######################################################################
def compute_analyticity(xf, is_time=False):
    """Determines analyticity based on presence of +/- bins. Outputs:

        - `1`  : analytic                   -- no positive bins
        - `-1` : anti-analytic              -- no negative bins
        - `2`  : analytic and anti-analytic -- only Nyquist and maybe dc
        - `0`  : non-analytic (has + and -, or has dc but no Nyquist)
        - `nan`: signal is zero
    """
    if is_time:
        xf = fft(xf)

    N = len(xf)
    eps = np.finfo(xf.dtype).eps * 10
    axf = np.abs(xf)

    has_dc = bool(axf[0] > eps)
    has_nyq = bool(axf[N//2] > eps)
    if has_nyq:
        out_if_has_no_pn = 2
    elif has_dc:
        out_if_has_no_pn = 0
    else:
        out_if_has_no_pn = np.nan

    if N == 1:
        out = 0
    elif N == 2:
        out = out_if_has_no_pn
    else:
        if N == 3:
            has_negatives = bool(axf[2] > eps)
            has_positives = bool(axf[1] > eps)
        else:
            has_negatives = bool(axf[N//2 + 1:].max() > eps)
            has_positives = bool(axf[1:N//2].max()    > eps)

        if has_negatives and has_positives:
            out = 0
        elif has_negatives and not has_positives:
            out = -1
        elif not has_negatives and has_positives:
            out = 1
        else:
            out = out_if_has_no_pn
    return out
