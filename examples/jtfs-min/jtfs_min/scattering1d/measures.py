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
import warnings
import math
import numpy as np
from scipy.fft import ifft


def compute_spatial_support(h_f, criterion_amplitude=1e-3, warn=False):
    """
    Computes the (half) temporal support of a family of centered,
    symmetric filters h provided in the Fourier domain
    This function computes the support N which is the smallest integer
    such that for all signals x and all filters h,
    \\| x \\conv h - x \\conv h_{[-N, N]} \\|_{\\infty} \\leq \\epsilon
        \\| x \\|_{\\infty}  (1)
    where 0<\\epsilon<1 is an acceptable error, and h_{[-N, N]} denotes the
    filter h whose support is restricted in the interval [-N, N]
    The resulting value N used to pad the signals to avoid boundary effects
    and numerical errors.
    If the support is too small, no such N might exist.
    In this case, N is defined as the half of the support of h, and a
    UserWarning is raised.
    Parameters
    ----------
    h_f : array_like
        a numpy array of size batch x time, where each row contains the
        Fourier transform of a filter which is centered and whose absolute
        value is symmetric
    criterion_amplitude : float, optional
        value \\epsilon controlling the numerical
        error. The larger criterion_amplitude, the smaller the temporal
        support and the larger the numerical error. Defaults to 1e-3
    warn: bool (default False)
        Whether to raise a warning upon `h_f` leading to boundary effects.
    Returns
    -------
    t_max : int
        temporal support which ensures (1) for all rows of h_f

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    filter_bank.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    if h_f.ndim == 2 and h_f.shape[0] > h_f.shape[1]:
        h_f = h_f.transpose(1, 0)
    elif h_f.ndim == 1:
        h_f = h_f[None]
    elif h_f.ndim > 2:
        raise ValueError("`h_f.ndim` must be <=2, got shape %s" % str(h_f.shape))
    if h_f.shape[-1] == 1:
        return 1

    h = ifft(h_f, axis=1)
    half_support = h.shape[1] // 2
    # check if any value in half of worst case of abs(h) is below criterion
    hhalf = np.max(np.abs(h[:, :half_support]), axis=0)
    max_amplitude = hhalf.max()
    meets_criterion_idxs = np.where(hhalf <= criterion_amplitude * max_amplitude
                                    )[0]
    if len(meets_criterion_idxs) != 0:
        # if it is possible
        N = meets_criterion_idxs.min() + 1
        # in this case pretend it's 1 less so external computations don't
        # have to double support since this is close enough
        if N == half_support:
            N -= 1
    else:
        # if there are none
        N = half_support
        if warn:
            # Raise a warning to say that there will be border effects
            warnings.warn('Signal support is too small to avoid border effects')
    return N


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
            pf, criterion_amplitude=criterion_amplitude)

        if N > 1e9:  # no-cov
            # avoid crash
            raise Exception("couldn't satisfy stop criterion before `N > 1e9`; "
                            "check `fn`")
        if pf_support < N or (max_N is not None and N > max_N):
            break
        N *= 2
    return N


def compute_max_dyadic_subsampling(xi, sigma, alpha=4.):
    """
    Computes the maximal dyadic subsampling which is possible for a Gabor
    filter of frequency xi and width sigma
    Finds the maximal integer j such that:
    omega_0 < 2^{-(j + 1)}
    where omega_0 is the boundary of the filter, defined as
    omega_0 = xi + alpha * sigma
    This ensures that the filter can be subsampled by a factor 2^j without
    aliasing.
    We use the same formula for Gabor and Morlet filters.
    Parameters
    ----------
    xi : float
        frequency of the filter in [0, 1]
    sigma : float
        frequential width of the filter
    alpha : float, optional
        parameter controlling the error done in the aliasing.
        The larger alpha, the smaller the error. Defaults to 4.
    Returns
    -------
    j : int
        integer such that 2^j is the maximal subsampling accepted by the
        Gabor filter without aliasing.

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    filter_bank.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    upper_bound = min(xi + alpha * sigma, 0.5)
    j = math.floor(-math.log2(upper_bound)) - 1
    j = int(j)
    return j
