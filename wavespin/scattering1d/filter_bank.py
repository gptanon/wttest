# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import numpy as np
import math
from scipy.fft import ifft
from ..utils.measures import (compute_spatial_support, compute_spatial_width,
                              compute_bandwidth, compute_bw_idxs,
                              compute_max_dyadic_subsampling)


def adaptive_choice_P(sigma, eps=1e-7):
    """
    Adaptive choice of the value of the number of periods in the frequency
    domain used to compute the Fourier transform of a Morlet wavelet.

    This function considers a Morlet wavelet defined as the sum
    of

      - a Gabor term $\hat \psi(\omega) = \hat g_{\sigma}(\omega - \\xi)$
        where 0 < xi < 1 is some frequency and g_{\sigma} is
        the Gaussian window defined in Fourier by
        $\hat g_{\sigma}(\omega) = e^{-\omega^2/(2 \sigma^2)}$
      - a low pass term $\hat \phi$ which is proportional to $\hat g_{\sigma}$.

    If sigma is too large, then these formula will lead to discontinuities
    in the frequency interval [0, 1] (which is the interval used by numpy.fft).
    We therefore choose a larger integer P >= 1 such that at the boundaries
    of the Fourier transform of both filters on the interval [1-P, P], the
    magnitude of the entries is below the required machine precision.
    Mathematically, this means we would need P to satisfy the relations:

        $|\\hat \\psi(P)| <= \eps$ and $|\\hat \\phi(1-P)| <= \eps$

    Since 0 <= xi <= 1, the latter implies the former. Hence the formula which
    is easily derived using the explicit formula for g_{\\sigma} in Fourier.

    Parameters
    ----------
    sigma: float
        Positive number controlling the bandwidth of the filters
    eps : float
        Positive number containing required precision. Defaults to 1e-7

    Returns
    -------
    P : int
        integer controlling the number of periods used to ensure the
        periodicity of the final Morlet filter in the frequency interval
        [0, 1[. The value of P will lead to the use of the frequency
        interval [1-P, P[, so that there are 2*P - 1 periods.

    References
    ----------
    From `kymatio/scattering1d/filter_bank.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    val = math.sqrt(-2 * (sigma**2) * math.log(eps))
    P = int(math.ceil(val + 1))
    return P


def fold_filter_fourier(h_f, nperiods=1, aggregation='sum'):
    """
    Subsample a filter in time by folding it in the Fourier domain.

        `ifft(h_f)[::k] == ifft(fold_filter_fourier(h_f, k))`

    Parameters
    ----------
    h_f : np.ndarray
        Array of length `N`.
    nperiods : int
        Subsampling factor. Called `nperiods` because this method is used
        only in context of periodization with `nperiods` periods.
    aggregation : str['sum', 'mean']
        `'sum'` will multiply subsampled time-domain signal by subsampling
        factor to conserve energy during scattering (rather not double-account
        for it since we already subsample after convolving).
        `'mean'` will only subsample the input.

    Returns
    -------
    v_f : array_like
        complex numpy array of size (M,), which is a periodization of
        h_f as described in the formula:
        v_f[k] = sum_{i=0}^{n_periods - 1} h_f[i * M + k]

    References
    ----------
        1. This is a modification of `kymatio/scattering1d/filter_bank.py` in
           https://github.com/kymatio/kymatio/blob/0.3.0/
           Kymatio, (C) 2018-present. The Kymatio developers.
        2. Explanations:

            - Visual & concise:
                https://dsp.stackexchange.com/a/87121/50076  (John Muradeli)
            - Derivations & filter applicability:
                https://dsp.stackexchange.com/a/74734/50076  (John Muradeli)
    """
    M = h_f.shape[0] // nperiods
    h_f_re = h_f.reshape(nperiods, M)
    v_f = (h_f_re.sum(axis=0) if aggregation == 'sum' else
           h_f_re.mean(axis=0))
    v_f = v_f if h_f.ndim == 1 else v_f[:, None]  # preserve dim
    return v_f


def morlet_1d(N, xi, sigma, normalize='l1', P_max=5, eps=1e-7,
              precision='double'):
    """
    Computes the Fourier transform of a Morlet filter.

    A Morlet filter is the sum of a Gabor filter and a low-pass filter
    to ensure that the sum has exactly zero mean in the temporal domain.
    It is defined by the following formula in time:
    psi(t) = g_{sigma}(t) (e^{i xi t} - beta)
    where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
    the cancelling parameter.

    Parameters
    ----------
    N : int
        size of the temporal support
    xi : float
        Peak center frequency, in [0, 1].
    sigma : float
        bandwidth parameter
    normalize : string
        normalization types for the filters. Defaults to `'l1'`.
        Supported are `'l1'` and `'l2'` (understood in time domain).
    P_max: int
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most `2*P_max - 1` periods
        are used, to ensure an equal distribution around 0.5). Defaults to `5`.
        Should be >= 1
    eps : float
        required machine precision (to choose the adequate `P`).
    precision : str
        `'single'` or `'double'`

    Returns
    -------
    morlet_f : array_like
        Numpy array of size `(N,)` containing the Fourier transform of the Morlet
        filter at the frequencies given by `np.fft.fftfreq(N)`.

    References
    ----------
    This is a modification of `kymatio/scattering1d/filter_bank.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    # arg checks
    if not isinstance(P_max, int):
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))

    # Find the adequate value of P
    # always do this at double precision since it's fast
    P = min(adaptive_choice_P(sigma, eps=eps), P_max)
    assert P >= 1

    # handle `dtype`
    dtype = {'single': np.float32, 'double': np.float64}[precision]
    cdtype = {'single': np.complex64, 'double': np.complex128}[precision]
    sigma = np.array(sigma).astype(dtype)
    xi = np.array(xi).astype(dtype)

    # Define the frequencies over [1-P, P)
    # == `arange((1 - P) * N, P * N) / N`
    freqs = np.linspace(1 - P, P, N * (2*P - 1), endpoint=False, dtype=dtype)
    if P == 1:
        # in this case, make sure that there is continuity around 0
        # by using the interval [-0.5, 0.5]
        freqs_low = np.fft.fftfreq(N).astype(dtype)
    elif P > 1:
        freqs_low = freqs

    # define the gabor at freq xi and the low-pass, both of width sigma
    gabor_f = np.exp(-(freqs - xi)**2 / (2 * sigma**2))
    low_pass_f = np.exp(-(freqs_low**2) / (2 * sigma**2))

    # subsample in time <=> fold in Fourier
    # We periodize by sampling a long enough interval in frequency to ensure we've
    # captured all of the wavelet, then folding to undo the excess sampling rate.
    # The result is a periodic copying of the continuous Fourier transform, whose
    # one period we sample.
    gabor_f = fold_filter_fourier(gabor_f, nperiods=2 * P - 1)
    low_pass_f = fold_filter_fourier(low_pass_f, nperiods=2 * P - 1)
    # find the summation factor to ensure that morlet_f[0] = 0.
    kappa = gabor_f[0] / low_pass_f[0]
    morlet_f = gabor_f - kappa * low_pass_f
    # normalize the Morlet if necessary
    morlet_f *= get_normalizing_factor(morlet_f, normalize=normalize)
    # cast to complex to speed up runtime
    morlet_f = morlet_f.astype(cdtype)
    return morlet_f


def get_normalizing_factor(h_f, normalize='l1'):
    """
    Computes the desired normalization factor for a filter defined in Fourier.

    Parameters
    ----------
    h_f : tensor
        Numpy vector containing the Fourier transform of a filter
    normalize : str
        Desired normalization type, either `'l1'` or `'l2'`. Defaults to 'l1'.

    Returns
    -------
    norm_factor : float
        such that `h_f * norm_factor` is the adequately normalized vector.

    References
    ----------
    This is a modification of `kymatio/scattering1d/filter_bank.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    h_t = ifft(h_f)
    ah_t = np.abs(h_t)
    asum = ah_t.sum()
    if asum < 1e-7:
        raise ValueError('Zero division error is very likely to occur, '
                         'aborting computations now.')
    normalize = normalize.split('-')[0]  # in case of `-energy`

    if normalize == 'l1':
        norm_factor = 1. / asum
    elif normalize == 'l2':
        norm_factor = 1. / np.sqrt(np.sum(ah_t**2))
    else:
        raise ValueError("`normalize` must be 'l1' or 'l2', got %s" % normalize)
    return norm_factor


def gauss_1d(N, sigma, normalize='l1', P_max=5, eps=1e-7, precision='double'):
    """
    Computes the Fourier transform of a low pass Gaussian window.

    \\hat g_{\\sigma}(\\omega) = e^{-\\omega^2 / 2 \\sigma^2}

    Parameters
    ----------
    N : int
        size of the temporal support
    sigma : float
        bandwidth parameter
    normalize : string
        Normalization type for the filters. Defaults to `'l1'`.
        Supported are `'l1'` and `'l2'` (understood in time domain).
    P_max : int
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most `2*P_max - 1` periods
        are used, to ensure an equal distribution around 0.5). Defaults to `5`.
        Should be >= 1
    eps : float
        required machine precision (to choose the adequate `P`)
    precision : str
        `'single'` or `'double'`

    Returns
    -------
    g_f : array_like
        Numpy array of size `(N,)` containing the Fourier transform of the
        filter at the frequencies given by `np.fft.fftfreq(N)`.

    References
    ----------
    This is a modification of `kymatio/scattering1d/filter_bank.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    # Find the adequate value of P
    if not isinstance(P_max, int):
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    P = min(adaptive_choice_P(sigma, eps=eps), P_max)
    assert P >= 1

    # handle `dtype`
    # first compute as real since it's faster, then cast to complex
    dtype = {'single': np.float32, 'double': np.float64}[precision]
    cdtype = {'single': np.complex64, 'double': np.complex128}[precision]
    sigma = np.array(sigma).astype(dtype)

    # switch cases
    if P == 1:
        freqs_low = np.fft.fftfreq(N).astype(dtype)
    elif P > 1:
        freqs_low = np.linspace(1 - P, P, N * (2*P - 1),
                                endpoint=False, dtype=dtype)
    # define the low pass over enough periods
    g_f = np.exp(-freqs_low**2 / (2 * sigma**2))
    # subsample it to the desired sampling rate
    g_f = fold_filter_fourier(g_f, nperiods=2 * P - 1)
    # normalize the signal
    g_f *= get_normalizing_factor(g_f, normalize=normalize)
    # cast to complex to speed up runtime
    g_f = g_f.astype(cdtype)
    # return the Fourier transform
    return g_f


def compute_sigma_psi(xi, Q, r_psi=math.sqrt(0.5)):
    """
    Computes the frequential width (in continuous-time) of a Morlet filter
    of frequency `xi`, belonging to a filterbank with `Q` wavelets per octave.

    This width, together with dependence on `xi` and `Q` that determines
    spacing between filters, controls the filters' intersection in frequency
    domain, measured as energy overlap (see `compute_filter_redundancy`).
    Ensures sufficient tiling of the whole frequency axis.

    Parameters
    ----------
    xi : float
        Frequency of the filter, in [0, 1].
    Q : int[>=1]
        Number of filters per octave.
    r_psi : float[>0, <1]
        Redundancy specification.
        The larger r_psi, the greater the overlap between adjacent filters,
        and the larger the returned value for the same `xi, Q`.
        Recommended to keep the default value.

    Returns
    -------
    sigma : float
        Frequential width of the Morlet wavelet, as a continuous-time parameter.

    References
    ----------
      1. Convolutional operators in the time-frequency domain, V. Lostanlen,
         PhD Thesis, 2017
         https://tel.archives-ouvertes.fr/tel-01559667
      2. This is a modification of `kymatio/scattering1d/filter_bank.py` in
         https://github.com/kymatio/kymatio/blob/0.3.0/
         Kymatio, (C) 2018-present. The Kymatio developers.
    """
    factor = 1. / 2**(1/Q)
    term1 = (1 - factor) / (1 + factor)
    term2 = 1. / math.sqrt(2 * math.log(1. / r_psi))
    return xi * term1 * term2


def move_one_dyadic_step(cv, Q):
    """
    Computes the parameters of the next wavelet on the low frequency side,
    based on the parameters of the current wavelet.

    This function is used in the loop defining all the filters, starting at the
    wavelet frequency and then going to the low frequencies by dyadic steps.

    The steps are defined as:

        xi_{n+1} = 2^{-1/Q} xi_n
        sigma_{n+1} = 2^{-1/Q} sigma_n

    Parameters
    ----------
    cv : dictionary
        stands for current_value. Is a dictionary with keys:

          - `'key'`: a tuple (j, n) where n is a counter and j is the maximal
            dyadic subsampling accepted by this wavelet.
          - `'xi`': central frequency of the wavelet
          - `'sigma'`: width of the wavelet
    Q : int
        Number of wavelets per octave. Controls the relationship between
        the frequency and width of the current wavelet and the next wavelet.

    Returns
    -------
    new_cv : dict
        Updated `cv`.

    References
    ----------
    This is a modification of `kymatio/scattering1d/filter_bank.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    factor = 1 / 2**(1/Q)
    n = cv['key']
    new_cv = {'xi': cv['xi'] * factor, 'sigma': cv['sigma'] * factor}
    new_cv['key'] = n + 1
    return new_cv


def compute_xi_max(Q):
    """
    Computes the maximal xi to use for the Morlet family, depending on Q.

    Larger `Q` <=> larger `xi_max`. Balances frequency tiling against analyticity:

        - lesser `Q` has greater bandwidth, so the highest-frequency wavelet
          spills more over into negatives.
        - larger `Q` has lesser bandwidth and we need to pack more wavelets,
          so we "start off" closer to Nyquist.

    Parameters
    ----------
    Q : int
        number of wavelets per octave (integer >= 1)

    Returns
    -------
    xi_max : float
        largest frequency of the wavelet frame.

    References
    ----------
    This is a modification of `kymatio/scattering1d/filter_bank.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    xi_max = max(0.4, 1. / (1. + 2**(1./Q)))
    return xi_max


def compute_params_filterbank(sigma_min, Q, r_psi=math.sqrt(0.5), J_pad=None):
    """
    Computes the parameters of a Morlet wavelet filterbank.

    This family is defined by constant ratios between the frequencies and
    bandwidths of adjacent filters, up to a minimum frequency where the frequencies
    are translated. `sigma_min` specifies the smallest frequential width among
    all filters, while preserving the coverage of the whole frequency axis.

    Parameters
    ----------
    sigma_min : float
        Acts as a lower-bound on the frequential widths of the band-pass
        filters. The low-pass filter may be wider (if `T < 2**J_scattering`),
        making invariants over shorter time scales than longest band-pass filter.
    Q : int
        Number of wavelets per octave.
    r_psi : float
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger `r_psi`, the larger the overlap between adjacent wavelets),
        and stability against time-warp deformations (larger `r_psi` improves it).
        Defaults to `sqrt(0.5)`.
    J_pad : int
        Used to compute `xi_min`, lower bound on `xi` to ensure every bandpass is
        a valid wavelet, i.e. doesn't peak below FFT(x_pad) bin 1. Else, we have

          - extreme distortion (can't peak between 0 and 1)
          - general padding may *introduce* new variability (frequencies) or
            cancel existing ones (e.g. half cycle -> full cycle), so we set
            `xi_min` w.r.t. padded rather than original input

        https://github.com/kymatio/kymatio/pull/737  # TODO

    Returns
    -------
    xi : list[float]
        list containing the central frequencies of the wavelets.
    sigma : list[float]
        list containing the frequential widths of the wavelets.
    is_cqt : list[bool]
        list containing True if a wavelet was built per Constant Q Transform
        (fixed `xi / sigma`), else False for the STFT portion.

    Meta info
    ---------
    Approximation (with `r_psi=sqrt(.5)`):

        n_cqt ~= Q * (J - log2(Q) + 0.57) + 1

    Exact:

        n_cqt = 1 + floor( Q*(ln(sigma_min) - ln(sigma_max)) / ln(0.5) )
        n_total = n_cqt + Q

        sigma_min = sigma0 / 2^J
        sigma_max = xi_max * (1-F)/(1+F) / sqrt(2 * ln(1 / r_psi))
        xi_max = max(0.4, 1 / (1 + 1/F))
        F = 0.5^(1/Q)

    Derived by solving s(n) = F^(n-1) * sigma_max, for y in s(y) = sigma_min.
    Approximation assumes r_psi=sqrt(.5), sigma0=0.13.

    Approximation derivation:

        smin := sigma_min
        C := ln(sigma_max)

        ln(smin) = ln(sigma0) - J*ln(2)

        ln(smin) - C =
        ln(sigma0) - J*ln(2) - C =
        K0 - J*ln(2)
            K0 = ln(sigma0) - C

        (ln(smin) - C) / ln(F) =
        (K0 - J*ln(2)) / ln(F) =
        (K0 + J*ln(0.5)) / ln(F) =
        (K0 + J*ln(0.5)) / ln(0.5) * Q =
        (K0/ln(0.5) + J) * Q =
        (K + J) * Q
            K = K0 / ln(0.5)

        -->

        n_cqt0 = Q * (J + K)
        n_cqt = 1 + floor(n_cqt0)

        `n_cqt0` is helpful to predict `n_cqt` from `J` and `Q`.
        Full reformulation:

        n_cqt0 =
        Q * (J + (ln(sigma0) - ln(sigma_max)) / ln(0.5)) ~=
        Q * (J + 2.94 + 1.44 * ln(sigma_max)) ~=
        Q * (J + 2.94 + 1.44 * ln(0.55 * (1 - F)/(1 + F)));

            1.44 * ln(0.55 * (1 - F)/(1 + F)) =
            1.44 * ( ln(0.55) + ln((1-F)/(1+F)) ) ~=
            -0.86 + 1.44 * ln((1-F)/(1+F))

            (1 - .5**(1/Q)) / (1 + .5**(1/Q)) --> series expansion -->
            .35/Q - .014/Q^3 + ... ~=
            .35/Q

            1.44 * ln(.35/Q) = 1.44*ln(.35) - 1.44*ln(Q) ~= -1.51 - 1.44*ln(Q)

        n_cqt0 ~= Q * (J + 2.94 - 0.86 - 1.51 - 1.44*ln(Q))
            ln(x) = log2(x) / log2(e) ~= 0.693 * log2(x)
            -1.44*ln(Q) ~= -1.443*0.693 * log2(Q) ~= -log2(Q)

        And so,

        n_cqt0 ~= Q * (J - log2(Q) + 0.57)

        hence,

            -----------------------------------------
            | n_cqt ~= Q * (J - log2(Q) + 0.57) + 1 |
            -----------------------------------------

        The approximation is excellent for Q >= 4, good for Q = 3, and okay for
        [1, 2]. More accurately, it takes floor. Assumes r_psi = sqrt(.5),
        sigma0 = 0.13.

        -----------------

        Approx for `last_xi`

        last_xi = xi_max * F**(n_cqt - 1)

        This is exact, so just plug in the approx for `n_cqt`.

    References
    ----------
      1. Convolutional operators in the time-frequency domain, V. Lostanlen,
         PhD Thesis, 2017
         https://tel.archives-ouvertes.fr/tel-01559667
      2. This is a modification of `kymatio/scattering1d/filter_bank.py` in
         https://github.com/kymatio/kymatio/blob/0.3.0/
         Kymatio, (C) 2018-present. The Kymatio developers.
    """
    # xi_min
    if J_pad is not None:
        # lowest peak at padded's bin 1
        xi_min = 1 / 2**J_pad
    else:
        # no limit
        xi_min = -1

    xi_max = compute_xi_max(Q)
    sigma_max = compute_sigma_psi(xi_max, Q, r_psi=r_psi)

    xi = []
    sigma = []
    is_cqt = []

    if sigma_max <= sigma_min or xi_max <= xi_min:
        # in this exceptional case, we will not go through the loop, so
        # we directly assign
        last_xi = xi_max
    else:
        # fill all the dyadic wavelets as long as possible
        current = {'key': 0, 'j': 0, 'xi': xi_max, 'sigma': sigma_max}
        # while we can attribute something
        while current['sigma'] > sigma_min and current['xi'] > xi_min:
            xi.append(current['xi'])
            sigma.append(current['sigma'])
            is_cqt.append(True)
            current = move_one_dyadic_step(current, Q)
        # get the last key
        last_xi = xi[-1]

    # fill num_intermediate wavelets between last_xi and xi_min, both excluded
    num_intermediate = Q
    for q in range(1, num_intermediate + 1):
        factor = (num_intermediate + 1. - q) / (num_intermediate + 1.)
        new_xi = factor * last_xi
        new_sigma = sigma_min

        xi.append(new_xi)
        sigma.append(new_sigma)
        is_cqt.append(False)
        if new_xi < xi_min:
            # break after appending one to guarantee tiling `xi_min`
            # in case `xi` increments are too small
            break
    # return results
    return xi, sigma, is_cqt


def calibrate_scattering_filters(J, Q, T, r_psi=math.sqrt(0.5), sigma0=0.13,
                                 J_pad=None):
    """
    Calibrates the parameters of the filters used at the 1st and 2nd orders
    of the scattering transform.

    These filterbanks share the same low-pass filter.

    Parameters
    ----------
    J : int
        See `wavespin.scattering1d.filter_bank.scattering_filter_factory`.
    Q : int / tuple[int]
        See `wavespin.scattering1d.filter_bank.scattering_filter_factory`.
    T : int
        See `wavespin.scattering1d.filter_bank.scattering_filter_factory`.
    r_psi : float / tuple[float]
        See `wavespin.scattering1d.filter_bank.scattering_filter_factory`.
    sigma0 : float
        See `wavespin.scattering1d.filter_bank.scattering_filter_factory`.
    xi_min : float
        Lower bound on `xi` to ensure every bandpass is a valid wavelet
        (doesn't peak at FFT bin 1) within `2*len(x)` padding.

    Returns
    -------
    sigma_low : float
        frequential width of the low-pass filter
    xi1s : list[float]
        Center frequencies of the first order filters.
    sigma1s : list[float]
        Frequential widths of the first order filters.
    is_cqt1s : list[bool]
        Constant Q Transform construction flag of the first order filters.
    xi2s, sigma2s, is_cqt2s :
        `xi1, sigma1, j1, is_cqt1` for second order filters.

    References
    ----------
    This is a modification of `kymatio/scattering1d/filter_bank.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    Q1, Q2 = Q if isinstance(Q, tuple) else (Q, 1)
    J1, J2 = J if isinstance(J, tuple) else (J, J)
    r_psi1, r_psi2 = r_psi if isinstance(r_psi, tuple) else (r_psi, r_psi)
    if Q1 < 1 or Q2 < 1:
        raise ValueError('Q should always be >= 1, got {}'.format(Q))

    # lower bound of band-pass filter frequential widths:
    # for default T = 2**(J), this coincides with sigma_low
    sigma_min1 = sigma0 / 2**J1
    sigma_min2 = sigma0 / 2**J2

    xi1s, sigma1s, is_cqt1s = compute_params_filterbank(
        sigma_min1, Q1, r_psi=r_psi1, J_pad=J_pad)
    xi2s, sigma2s, is_cqt2s = compute_params_filterbank(
        sigma_min2, Q2, r_psi=r_psi2, J_pad=J_pad)

    # width of the low-pass filter
    sigma_low = sigma0 / T
    return sigma_low, xi1s, sigma1s, is_cqt1s, xi2s, sigma2s, is_cqt2s


def scattering_filter_factory(N, J_support, J_scattering, Q, T,
                              r_psi=math.sqrt(0.5), criterion_amplitude=1e-3,
                              normalize='l1', analytic=False, sigma0=0.13,
                              P_max=5, eps=1e-7, precision='double'):
    """
    Builds in Fourier domain the Morlet wavelets and Gaussian lowpass filters
    used for the scattering transform.

    Each filter is provided as a dictionary with "meta" information; see below.
    Most importantly, filters themselves are keyed by integers `k`, where
    `2**k` is their subsampling factor.

    Parameters
    ----------
    N : int
        Length of input signals.

    J_support : int
        Padded length. `2**J_support` = length of the filters.

    J_scattering : int
        `2**J_scattering` = maximum temporal width of any filter.

    Q : int >= 1 / tuple[int]
        Controls number of wavelets per octave and time-frequency resolution.
        See `help(wavespin.Scattering1D())`.

    T : int
        Temporal width of the low-pass filter.
        See `help(wavespin.Scattering1D())`.

    r_psi : float[>0, <1] / tuple[float]
        Wavelet redundancy.
        See `help(wavespin.Scattering1D())`.

    criterion_amplitude : float
        See `help(wavespin.Scattering1D())`.

    normalize : string / tuple[string]
        Normalization mode for the filters (in the temporal domain).
        Supports `'l1'`, `'l2'`, `'l1-energy'`, `'l2-energy'`, but only
        `'l1'` or `'l2'` is used here. See `help(wavespin.Scattering1D())`.

    analytic : bool (default False)
        Whether to enforce strict analyticity by zeroing negative frequencies.
        See `help(wavespin.Scattering1D())`.

    sigma0 : float
        See `help(wavespin.Scattering1D())`.

    P_max : int >= 1
        See `help(wavespin.Scattering1D())`.

    eps : float
        See `help(wavespin.Scattering1D())`.

    precision : str
        `'single'` or `'double'`

    Returns
    -------
    phi_f : dict
        Low-pass filter at all possible subsamplings. The possible subsamplings
        are controlled by the inputs they can receive, which correspond to the
        subsamplings performed on top of the 1st and 2nd order transforms.
        The dict also stores meta, e.g. `phi_f['xi']`, see "Meta" below.
    psi1_f : list[dict]
        Band-pass filters of the 1st order, only for the base resolution as no
        subsampling is used in the scattering tree.
        Each element of this list is a dict that contains a filter, its
        subsamplings, and meta (e.g. `psi1_f[0]['xi']`, see "Meta" below).
    psi2_f : list[dict]
        Band-pass filters of the 2nd order at all possible subsamplings. The
        subsamplings are determined by the input they can receive, which depends
        on the scattering tree.
        Each element of this list is a dict that contains a filter, its
        subsamplings, and meta (e.g. `psi2_f[0]['xi']`, see "Meta" below).

    Meta
    ----
    - `'xi'`: float
       Center frequency as a continuous-time parameter, defaults to `0.` for
       low-pass filters. Also see `'peak_idx'`.

    - `'sigma`': float:
       Bandwidth as a continuous-time parameter. Not a reliable indicator
       of true bandwidth (see `'bw'`).

    - `k`: int >= 0
       Dyadic subsampling factors (in time).
       E.g. `phi_f[2]` stores the Fourier transform of the lowpass filter
       after it's been subsampled in time by `2**2`.

    - `'j'`: int >= 0
       Maximal value of k. Set such that the energy aliased upon subsampling
       is `criterion_amplitude**2` of the filter's total energy.

    - `'is_cqt'`: bool
       Whether the filter is part of the CQT portion of the filterbank
       (Constant-Q Transform, `Q=xi/sigma`, where Q=quality factor, not to
       be confused with the `Q` parameter).

    - `'width'`: int
       Temporal width, in number of samples
       (interval of temporal invariance, i.e. its "T"). See `sigma0`, and
       https://wavespon.readthedocs.io/en/latest/extended/general_method_docs.html

    - `'support'`: int
       Temporal support, in number of samples
       (interval outside of which filter is ~0 in time). See `sigma0`, and
       https://wavespon.readthedocs.io/en/latest/extended/general_method_docs.html

    - `'scale'`: int
       Temporal dyadic scale, in number of samples (`=ceil(log2(support))`).
       Scale of scattering (convolution intervals), rather than scale of
       invariance. See `sigma0`.

    - `'bw'`: int
       Bandwidth, in number of samples
       (interval outside of which filter is ~0 in frequency).

         - Measures true (realized) bandwidth, unlike `'sigma'` which doesn't
           account for insufficient decay and Morlet's correction term for
           low center frequencies.
         - "Invariance" amount is actually defined in terms of bandwidth
           of the lowpass filter: `T == len(phi) / bw(phi)`.
           This follows a secondary meaning of "invariance": `T` means we
           can losslessly subsample by `T`, so fewer samples represent the
           same variation: x16 subsampling means, pre-subsampling, 16
           samples don't represent any variation that 1 sample can't.

    - `'bw_idxs'`: tuple[int]
       Indices of frequential support
       (interval outside of which filter is ~0 in frequency).

           - It's `'bw'`, but in indices, also measured a little differently.
           - The intent is described in `smart_paths_exclude`.
           - Excluded for lowpass since it serves no purpose and is ill-defined
             (the indices are meant to slice the filter, though we could
             redefine).

    - `'peak_idx'`: int
       Center frequency, as index of the maximum of absolute value
       (e.g. `np.argmax(np.abs(psi1_f))`).

           - `'sigma'` <=> `'bw'`, `'xi'` <=> `'peak_idx'`.
           - Defined support-inclusive. That is, `psi1_f[left:right + 1]`
             slices the "support" interval.

    References
    ----------
      1. Convolutional operators in the time-frequency domain, V. Lostanlen,
         PhD Thesis, 2017
         https://tel.archives-ouvertes.fr/tel-01559667
      2. This is a modification of `kymatio/scattering1d/filter_bank.py` in
         https://github.com/kymatio/kymatio/blob/0.3.0/
         Kymatio, (C) 2018-present. The Kymatio developers.
    """
    # compute the spectral parameters of the filters
    (sigma_low, xi1s, sigma1s, is_cqt1s, xi2s, sigma2s, is_cqt2s
     ) = calibrate_scattering_filters(J_scattering, Q, T, r_psi=r_psi,
                                      sigma0=sigma0, J_pad=J_support)

    # split `normalize` into orders
    if isinstance(normalize, tuple):
        normalize1, normalize2 = normalize
    else:  # no-cov
        normalize1 = normalize2 = normalize

    # instantiate the dictionaries which will contain the filters and j metas
    phi_f = {}
    psi1_f, psi2_f = [], []
    metas1, metas2 = [{field: [] for field in ('j', 'peak_idx', 'bw_idxs')}
                      for _ in range(2)]

    # first compute unsubsampled filters to determine their j's, along meta
    # necessary to avoid recomputation
    N_filters = 2**J_support
    params_orders = [(psi1_f, metas1, normalize1, xi1s, sigma1s),
                     (psi2_f, metas2, normalize2, xi2s, sigma2s)]
    for order_idx, params in enumerate(params_orders):
        psi_fs, metas, normalize, xis, sigmas = params

        for n, (xi, sigma) in enumerate(zip(xis, sigmas)):
            pf = morlet_1d(N_filters, xi, sigma, normalize=normalize,
                           P_max=P_max, eps=eps, precision=precision)
            peak_idx = np.argmax(np.abs(pf))
            bw_idxs = compute_bw_idxs(pf, criterion_amplitude, c=peak_idx)
            if analytic:  # adjust manually instead of recomputing
                bw_idxs = (bw_idxs[0], min(bw_idxs[1], N_filters//2))
            j = compute_max_dyadic_subsampling(pf, bw_idxs)

            psi_fs.append({0: pf})
            meta = {'j': j, 'peak_idx': peak_idx, 'bw_idxs': bw_idxs}
            for k, v in meta.items():
                metas[k].append(v)

    # fetch `j`s for brevity
    j1s, j2s = metas1['j'], metas2['j']

    # compute second-order subsamplings
    for (n2, j2) in enumerate(j2s):
        # compute the current value for the max subsampling, which depends on
        # the input it can accept.
        # if j2=5 but max j1 is 3, no need to compute subsamplings 4 and 5.
        max_sub_psi2 = min(max(j1s), j2)

        # compute the filter after subsampling at all other subsamplings
        # which might be received by the network, based on this first filter
        for subsampling in range(1, max_sub_psi2 + 1):
            factor_subsampling = 2**subsampling
            psi2_f[n2][subsampling] = fold_filter_fourier(
                psi2_f[n2][0], nperiods=factor_subsampling)

    # for the 1st order filters, the input is not subsampled so we
    # can only compute them with N_pad=2**J_support

    # compute the low-pass filters phi
    # Determine the maximal subsampling for phi, which depends on the
    # input it can accept (both 1st and 2nd order)
    log2_T = math.floor(math.log2(T))
    max_subsampling_after_psi1 = max(j1s)
    max_subsampling_after_psi2 = max(j2s)
    max_sub_phi = min(log2_T, max(max_subsampling_after_psi1,
                                  max_subsampling_after_psi2))

    # compute the filters at all encountered subsamplings
    phi_f[0] = gauss_1d(N_filters, sigma_low, normalize=normalize1,
                        P_max=P_max, eps=eps, precision=precision)
    for subsampling in range(1, max_sub_phi + 1):
        phi_f[subsampling] = fold_filter_fourier(
            phi_f[0], nperiods=2**subsampling)

    # Embed the meta information within the filters ##########################
    fields = ('width', 'support', 'scale', 'bw', 'bw_idxs', 'peak_idx')
    ca = dict(criterion_amplitude=criterion_amplitude)
    s0ca = dict(N=N, sigma0=sigma0, criterion_amplitude=criterion_amplitude)

    for (n1, j1) in enumerate(j1s):
        pf = psi1_f[n1][0]
        # handle strict analyticity
        if analytic:
            make_strictly_analytic(pf)

        peak_idx = metas1['peak_idx'][n1]
        bw_idxs = metas1['bw_idxs'][n1]
        width = compute_spatial_width(pf, **s0ca)

        psi1_f[n1]['xi'] = xi1s[n1]
        psi1_f[n1]['sigma'] = sigma1s[n1]
        psi1_f[n1]['j'] = j1
        psi1_f[n1]['is_cqt'] = is_cqt1s[n1]
        psi1_f[n1]['width'] = {0: width}
        psi1_f[n1]['support'] = {0: compute_spatial_support(pf, **ca)}
        psi1_f[n1]['scale'] = {0: width_to_scale(width)}
        psi1_f[n1]['bw'] = {0: compute_bandwidth(pf, **ca, c=peak_idx)}
        psi1_f[n1]['bw_idxs'] = {0: bw_idxs}
        psi1_f[n1]['peak_idx'] = {0: peak_idx}

    for (n2, j2) in enumerate(j2s):
        psi2_f[n2]['xi'] = xi2s[n2]
        psi2_f[n2]['sigma'] = sigma2s[n2]
        psi2_f[n2]['j'] = j2
        psi2_f[n2]['is_cqt'] = is_cqt2s[n2]

        for field in fields:
            psi2_f[n2][field] = {}
        for k in psi2_f[n2]:
            if isinstance(k, int):
                pf = psi2_f[n2][k]
                if analytic:
                    make_strictly_analytic(pf)

                if k == 0:
                    peak_idx = metas2['peak_idx'][n2]
                    bw_idxs = metas2['bw_idxs'][n2]
                else:
                    peak_idx = np.argmax(np.abs(pf))
                    bw_idxs = compute_bw_idxs(pf, **ca, c=peak_idx)
                width = compute_spatial_width(pf, **s0ca)

                psi2_f[n2]['width'][k] = width
                psi2_f[n2]['support'][k] = compute_spatial_support(pf, **ca)
                psi2_f[n2]['scale'][k] = width_to_scale(width)
                psi2_f[n2]['bw'][k] = compute_bandwidth(pf, **ca, c=peak_idx)
                psi2_f[n2]['bw_idxs'][k] = bw_idxs
                psi2_f[n2]['peak_idx'][k] = peak_idx

    peak_idx = 0
    phi_f['xi'] = 0.
    phi_f['sigma'] = sigma_low
    phi_f['j'] = log2_T

    for field in fields:
        phi_f[field] = {}
    for k in phi_f:
        if isinstance(k, int):
            pf = phi_f[k]
            width = compute_spatial_width(pf, **s0ca)

            phi_f['width'][k] = width
            phi_f['support'][k] = compute_spatial_support(pf, **ca)
            phi_f['scale'][k] = width_to_scale(width)
            phi_f['bw'][k] = compute_bandwidth(pf, **ca, c=peak_idx)
            phi_f['bw_idxs'][k] = compute_bw_idxs(pf, **ca, c=peak_idx)
            phi_f['peak_idx'][k] = peak_idx

    # sanity check: aliasing
    for n1, p1f in enumerate(psi1_f):
        peak_idx1, j1 = p1f['peak_idx'][0], p1f['j']
        assert peak_idx1 <= N_filters // 2 // 2**j1, (
            peak_idx1, N_filters, j1)

    # return results
    return phi_f, psi1_f, psi2_f

#### misc ####################################################################
def make_strictly_analytic(pf, anti_analytic=False):
    # edge cases
    if pf.size in (1, 2):
        return pf

    # ensure it's only one filter
    assert pf.ndim == 1 or max(pf.shape) == pf.size, pf.shape

    N = len(pf)
    if anti_analytic:
        pf[1:N//2] = 0      # zero positives
    else:
        pf[N//2 + 1:] = 0   # zero negatives
    pf[N//2] /= 2           # halve Nyquist
    # Nyquist is halved consistently with the Hilbert transform, also
    # improving temporal decay: https://github.com/jonathanlilly/jLab/issues/13


def width_to_scale(width):
    """width -> scale"""
    return math.ceil(math.log2(width))


def N_and_pad_to_J_pad(N, min_to_pad):
    """N, min_to_pad -> J_pad"""
    return math.ceil(math.log2(N + min_to_pad))


def j2_j1_cond(j2, j1):
    """`j`-criterion to keep an `n2`-`n1` pair. `False` means exclude.

    On forbidding `j2 == 0`
    -----------------------
    `j2 != 0` originated for consistency with the earler `j2 > j1` criterion
    (which made `j2 == 0` impossible), and kept

       - to avoid compute expense (esp. for JTFS)
       - since it's practically highly unlikely to generate qualifying signals,
         even if `Q1` is low (and especially since we almost always want `>=8`)
       - since with the typical `Q2=1`, the `j2==0` wavelets are low quality

    Not necessarily ideal to not allow overriding, but the idea's not been
    explored.
    """
    return j2 != 0 and j2 >= j1


def n2_n1_cond(n1, n2, j1, j2, paths_exclude):
    """Complete condition to keep an `n2`-`n1` pair. `False` means exclude.

    `not (not A or B or C)` == `A and not B and not C`.
    Code does former for performance.
    """
    return not (not j2_j1_cond(j2, j1) or
                n2 in paths_exclude.get('n2', []) or
                (n2, n1) in paths_exclude.get('n2, n1', []))
