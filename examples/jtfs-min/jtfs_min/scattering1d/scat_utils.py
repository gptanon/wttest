# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import math

from .filter_bank import calibrate_scattering_filters, gauss_1d, morlet_1d
from .measures import compute_spatial_support, compute_minimum_required_length


def compute_border_indices(log2_T, J, i0, i1):
    """
    Computes border indices at all scales which correspond to the original
    signal boundaries after padding.

    At the finest resolution,
    original_signal = padded_signal[..., i0:i1].
    This function finds the integers i0, i1 for all temporal subsamplings
    by 2**J, being conservative on the indices.

    Maximal subsampling is by `2**log2_T` if `average=True`, else by
    `2**max(log2_T, J)`. We compute indices up to latter to be sure.

    Parameters
    ----------
    log2_T : int
        Maximal subsampling by low-pass filtering is `2**log2_T`.
    J : int / tuple[int]
        Maximal subsampling by band-pass filtering is `2**J`.
    i0 : int
        start index of the original signal at the finest resolution
    i1 : int
        end index (excluded) of the original signal at the finest resolution

    Returns
    -------
    ind_start, ind_end: dictionaries with keys in [0, ..., log2_T] such that the
        original signal is in padded_signal[ind_start[j]:ind_end[j]]
        after subsampling by 2**j

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/utils.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    if isinstance(J, tuple):
        J = max(J)
    ind_start = {0: i0}
    ind_end = {0: i1}
    for j in range(1, max(log2_T, J) + 1):
        ind_start[j] = math.ceil(ind_start[j - 1] / 2)
        ind_end[j] = math.ceil(ind_end[j - 1] / 2)
    return ind_start, ind_end


def compute_padding(J_pad, N):
    """
    Computes the padding to be added on the left and on the right
    of the signal.

    It should hold that 2**J_pad >= N

    Parameters
    ----------
    J_pad : int
        2**J_pad is the support of the padded signal
    N : int
        original signal support size

    Returns
    -------
    pad_left: amount to pad on the left ("beginning" of the support)
    pad_right: amount to pad on the right ("end" of the support)

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/utils.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    N_pad = 2**J_pad
    if N_pad < N:
        raise ValueError('Padding support should be larger than the original '
                         'signal size!')
    to_add = 2**J_pad - N
    pad_right = to_add // 2
    pad_left = to_add - pad_right
    return pad_left, pad_right


def compute_minimum_support_to_pad(N, J, Q, T, criterion_amplitude=1e-3,
                                   normalize='l1', r_psi=math.sqrt(0.5),
                                   sigma0=.13, P_max=5, eps=1e-7):
    """
    Computes the support to pad given the input size and the parameters of the
    scattering transform.

    Parameters
    ----------
    N : int
        temporal size of the input signal

    J : int
        scale of the scattering

    Q : int >= 1
        The number of first-order wavelets per octave. Defaults to `1`.
        If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
        second-order wavelets per octave (which defaults to `1`).

          - If `Q1==0`, will exclude `psi1_f` from computation.
          - If `Q2==0`, will exclude `psi2_f` from computation.

    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling

    normalize : string / tuple[string]
        Normalization convention for the filters (in the temporal domain).
        Supports 'l1', 'l2', 'l1-energy', 'l2-energy', but only 'l1' or 'l2' is
        used. See `help(Scattering1D)`.

    criterion_amplitude: float `>0` and `<1`
        Represents the numerical error which is allowed to be lost after
        convolution and padding.
        The larger `criterion_amplitude`, the smaller the padding size is.
        Defaults to `1e-3`

    r_psi : float
        See `help(wavespin.Scattering1D())`.

    sigma0 : float
        See `help(wavespin.Scattering1D())`.

    P_max : int
        See `help(wavespin.Scattering1D())`.

    eps : float
        See `help(wavespin.Scattering1D())`.

    Returns
    -------
    min_to_pad: int
        Minimal value to pad the signal to avoid boundary effects and insufficient
        filter decay.
    """
    # compute params for calibrating, & calibrate
    Q1, Q2 = Q if isinstance(Q, tuple) else (Q, 1)
    Q_temp = (max(Q1, 1), max(Q2, 1))  # don't pass in zero
    N_init = N

    # `None` means `xi_min` is limitless. Since this method is used to compute
    # padding, then we can't know what it is, so we compute worst case.
    # If `max_pad_factor=None`, then the realized filterbank's (what's built)
    # `xi_min` is also limitless. Else, it'll be greater, depending on
    # `max_pad_factor`.
    J_pad = None
    sigma_low, xi1, sigma1, _, xi2, sigma2, _ = calibrate_scattering_filters(
        J, Q_temp, T, r_psi=r_psi, sigma0=sigma0, J_pad=J_pad)

    # split `normalize` into orders
    if isinstance(normalize, tuple):
        normalize1, normalize2 = normalize
    else:
        normalize1 = normalize2 = normalize

    # compute psi1_f with greatest time support, if requested
    if Q1 >= 1:
        psi1_f_fn = lambda N: morlet_1d(
            N, xi1[-1], sigma1[-1], normalize=normalize1, P_max=P_max, eps=eps)
    # compute psi2_f with greatest time support, if requested
    if Q2 >= 1:
        psi2_f_fn = lambda N: morlet_1d(
            N, xi2[-1], sigma2[-1], normalize=normalize2, P_max=P_max, eps=eps)
    # compute lowpass
    phi_f_fn = lambda N: gauss_1d(N, sigma_low, normalize=normalize1,
                                  P_max=P_max, eps=eps)

    # compute for all cases as psi's time support might exceed phi's
    ca = dict(criterion_amplitude=criterion_amplitude)
    N_min_phi = compute_minimum_required_length(phi_f_fn, N_init=N_init, **ca)
    phi_support = compute_spatial_support(phi_f_fn(N_min_phi), **ca)

    if Q1 >= 1:
        N_min_psi1 = compute_minimum_required_length(psi1_f_fn, N_init=N_init,
                                                     **ca)
        psi1_support = compute_spatial_support(psi1_f_fn(N_min_psi1), **ca)
    else:
        psi1_support = -1  # placeholder
    if Q2 >= 1:
        N_min_psi2 = compute_minimum_required_length(psi2_f_fn, N_init=N_init,
                                                     **ca)
        psi2_support = compute_spatial_support(psi2_f_fn(N_min_psi2), **ca)
    else:
        psi2_support = -1

    # set min to pad based on each
    pads = (phi_support, psi1_support, psi2_support)
    pad_phi, pad_psi1, pad_psi2 = pads

    # set main quantity as the max of all
    min_to_pad = max(pads)

    # return results
    return min_to_pad, pad_phi, pad_psi1, pad_psi2
