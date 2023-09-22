# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import numpy as np
import math

from .filter_bank import calibrate_scattering_filters, gauss_1d, morlet_1d
from ..utils.measures import (compute_spatial_support,
                              compute_minimum_required_length)


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
    This is a modification of `kymatio/scattering1d/utils.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
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

    It should hold that `2**J_pad >= N`

    Parameters
    ----------
    J_pad : int
        `2**J_pad` is the length of the padded signal.
    N : int
        Original signal length.

    Returns
    -------
    pad_left: amount to pad on the left ("beginning" of the support)
    pad_right: amount to pad on the right ("end" of the support)

    References
    ----------
    This is a modification of `kymatio/scattering1d/utils.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    N_pad = 2**J_pad
    if N_pad < N:
        raise ValueError('Padding length should be larger than the original '
                         'signal size!')
    to_add = 2**J_pad - N
    pad_right = to_add // 2
    pad_left = to_add - pad_right
    return pad_left, pad_right


def compute_minimum_support_to_pad(N, J, Q, T, criterion_amplitude=1e-3,
                                   normalize='l1', r_psi=math.sqrt(0.5),
                                   sigma0=.13, P_max=5, eps=1e-7,
                                   pad_mode='reflect', halve_zero_pad=True):
    """
    Computes the support to pad given the input size and the parameters of the
    scattering transform.

    Parameters
    ----------
    N : int
        `len(x)`, length of input to the transform.

    J : int
        See `help(wavespin.Scattering1D())`.

    Q : int >= 1
        See `help(wavespin.Scattering1D())`.

          - If `Q1==0`, will exclude `psi1_f` from computation.
          - If `Q2==0`, will exclude `psi2_f` from computation.

    T : int
        See `help(wavespin.Scattering1D())`.
        Controls pad contribution due to lowpass filter.

    normalize : string / tuple[string]
        See `help(wavespin.Scattering1D())`.
        The `-energy` specs have no effect here.

    criterion_amplitude: float `>0` and `<1`
        See `help(wavespin.Scattering1D())`.
        Controls the measure of "support", which is used to compute pad amounts.

    r_psi : float
        See `help(wavespin.Scattering1D())`.

    sigma0 : float
        See `help(wavespin.Scattering1D())`.

    P_max : int
        See `help(wavespin.Scattering1D())`.

    eps : float
        See `help(wavespin.Scattering1D())`.

    pad_mode : str
        See `help(wavespin.Scattering1D())`.

    halve_zero_pad : bool
        See `help(wavespin.Scattering1D())`.

    Returns
    -------
    min_to_pad: int
        Minimal value to pad the signal to avoid boundary effects and insufficient
        filter decay.

    Limitations
    -----------
    In summary, the pad logic isn't entirely correct, but is sufficient for
    `pad_mode` that's `'zero'` or `'reflect'`.

    Logic here is only complete up to the unaveraged first order. With lowpassing
    or higher orders, greater padding is required. This logic is reused through
    the entire library. The resulting errors are small or tiny, and practically
    acceptable, but sometimes not so in idealized theoretical settings.
    As explained below in "extent of error", the error should be very small for
    `'zero'` and `'reflect`' pads.

    This was discovered late in development and there was no time to account for
    all the implications. Workarounds:

        - `'zero'` and `'reflect'` for `pad_mode` should work fine
        - Manually padding and unpadding the input, and passing it as `x`, in
          addition to the padding done internally
        - Taking `J <= log2(N) - 4` and `T <= N / 2**4`
        - `pad_mode='zero'` and `wavespin.CFG['S1D']['halve_zero_pad'] = False`,
          and equivalently for JTFS.

    Limitations - explanation
    -------------------------
    It's due to temporal expansiveness of convolution. "Success" here is avoiding
    boundary effects due to circular convolution - meaning, the unpadded output
    matches the result of direct convolution. This means "left doesn't draw
    from right", and vice versa.

    Consider `S1` as follows, discarding modulus for clarity:

        1. `supp(psi1) = 50`, `supp(phi) = 60`, `len(x) = 100`.
           `supp` := support. Suppose `len(x) = supp(x)` for generality.
        2. `supp(conv(psi1, x)) = supp(psi1) + supp(x) = 150`.
        3. Next is lowpassing. Suppose `x` was padded to 500, and the original
           spans from index 200 to 300.
           Then, for "success", we require that `phi` at 300 doesn't circularly
           draw from left, and at 200 from right. Focus on the 300.
        4. `phi` centered at 300 spans [270, 330].
           Hence, the output of `conv(psi1, x)` must not draw from left for all
           those points, meaning up to 330. For that to happen, we repeat the
           analysis with `psi1` centered at 330: its span there is
           [305, 355].
           Thus, the required padding is up to 355. Total pad is 110.
        5. This method computes padding from the right as
           `max(supp(psi1)/2, supp(phi)/2)`.
           Yet as we saw, it should actually be
           `supp(psi1)/2 + supp(phi)/2`.

    To generalize, we consider the necessary chain of supports for all stacks of
    convolutions up to the unpadded output. Continued for `S2`:

        6. Let `supp(psi2) = 70`.
        7. `phi` centered at 300 spans [270, 330].
           Hence, the output of `conv(psi2, conv(psi1, x))` must not draw from
           left for all those points, meaning up to 330. For that to happen, we
           repeat the analysis with `psi2` centered at at 330: its span there is
           [295, 365].
           Hence, the output of `conv(psi1, x)` must not draw from left for all
           those points, meaning up to 365. For that to happen, we repeat the
           analysis with `psi1` centered at 365: its span there is
           [340, 390].
           Thus, the required padding is up to 390.
        8. This method computes padding from the right as
           `max(supp(psi1)/2, supp(psi2)/2, supp(phi)/2)`.
           Yet as we saw, it should actually be
           `supp(psi1)/2 + supp(psi2)/2 + supp(phi)/2`.

    Hence, for total padding, it's the sum of all convolutional supports up to
    the output. An exception is `pad_mode='zero'`, shown in a below section.

    Limitations: extent of error
    ----------------------------
    To get an idea for the extent of error of the `max`-based computation used
    by this function, we observe that, although in step 4 we require up to 355,
    the contribution of `psi1` at 330 to `phi` at 300 is minimal. Visually, in
    the ideally padded case, the peak of `psi1` is at the tail of `phi`.

    However, the error is reduced much further, by luck, in the two built-in
    schemes, `'reflect'` and `'zero'`:

        - `'reflect'`: if the input length is `Np2 + 1`, where `Np2` is a power
          of 2, and we pad to any higher power of 2, then the result is circularly
          continuous - meaning, `x_pad[0]` comes after `x_pad[-1]`, just like
          `x[6]` would come after `x[5]`. For simply `Np2`, it's close enough
          for all practical purposes (still, we can elect to make it exact).
          This also halves the pad requirement, except if we did half it we'd
          no longer have circular continuity, hence we don't - but the benefits
          still apply. The math in preceding sections doesn't account for this,
          and would reduce pad requirements.
        - `'zero'`: as shown in a below section, the pad requirements in this
          case are less than directly additive in filter supports. There's also
          of course the indifference to "left" and "right" over padded regions,
          since it's all zeros, which halves the pad requirements, but that's
          already accounted for in the function.

    The error is reduced yet further, by more luck, per `sigma0`, which yields
    filter `'support'` that's just barely above a power of 2 for any `J` or
    dyadic `T` - meaning in `max_pad_factor=None` case we add an extra power of
    2 to padding just for sake of a few samples. This may suffice to make the
    padding strictly correct, but it wasn't confirmed.

    Lastly, there's experimental evidence that the error's indeed tiny, in
    `tests/scattering1d/test_jtfs.py`; a few tests rely heavily on absence of
    boundary effects to satisfy difficult numeric assertions.

    Limitations: `pad_mode='zero'` case
    -----------------------------------
    We modify as follows:

        4b. `phi` centered at 300 spans [270, 330].
           Hence, the output of `conv(psi1, x)` must not draw from left for all
           those points, meaning up to 330. For that to happen, we repeat the
           analysis with `psi1` centered at 330: its span there is
           [305, 355]. However, we note that the span [301, 355] is identical
           to the span [0, 54], and to [144, 199]: all zeros. Without changing
           the indexing reference, we proceed to change padding to bare minimum
           required, on the fly.
           Begin by determining the span of `conv(psi1, x)`. That'll be
           [200-25, 300+25] = [175, 325].
           For `phi` at right, we're concerned with `psi1` at 330; for `psi1` at
           330 to not draw from 175, we pad to [175, 355] (chose right for
           clarity).
           For `phi` at left,  we're concerned with `psi1` at 170; for `psi1` at
           170 to not draw from 325, we would pad to [145, 325], or equivalently
           [175, 355] if not for 170, so [170, 350] - but we already did,
           take [170, 350]: `psi1` at 170 can only see up to 326 (1 right of 325),
           and `psi1` at 330 can only see up to 350 (1 circ-left of 170).
           Total pad is (200-170) + (350-300) = 80.
        5b. `supp(psi1) + supp(phi)/2` is total pad.
           The result for `supp(psi1) > supp(phi)` wasn't investigated and
           may be different, but it won't exceed the `pad_mode!='zero'` case.

    `S2` isn't investigated but we note the following: the extent of
    `conv(psi2, conv(psi1, x))` isn't based on `supp(psi1) + supp(psi2)`,
    but rather on `supp(conv(psi1, psi2))`, which is substantially different.

    Limitations: fundamental
    ------------------------
    When we say "eliminate boundary effects", we don't mean all of them. There's
    only two combinations that eliminate all boundary effects:

        1. Input is time-limited. That is, it's actually zero outside of the
           specified array.
        2. Padding is `'zero'` and sufficiently long. Meaning, (a) all circular
           effects are avoided, and all filters are unaliased (decay sufficiently
           in both time and frequency domains).
        3. No unpadding is done. Alternatively, we unpad based on the `'support'`
           of the *output*. Unpadding is aliasing and lossy; the complete
           wavelet transform in time-expansive per Heisenberg's uncertainty.
           This also concerns energy conservation:
           https://dsp.stackexchange.com/a/86182/50076

    By "time-limited" and "unaliased" and "eliminates", we mean that the said
    effects are within specified tolerances, all of which are controlled by
    `criterion_amplitude`. "Truly exact" requires infinitely long arrays and
    infinite machine precision, which is pointless as real-world SNR drowns out
    numeric errors.

    One might say there's a second combination, namely one where the actual
    larger signal is used as padding - but point `3` applies. More importantly,
    the point of doing these transforms is obtaining high-quality features, and
    not unpadding would include mostly zeros, which works against this goal by
    reducing information density.
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

    # can pad half as much
    if pad_mode == 'zero' and halve_zero_pad:
        pads = [p//2 for p in pads]
    pad_phi, pad_psi1, pad_psi2 = pads
    # set main quantity as the max of all
    min_to_pad = max(pads)

    # return results
    return min_to_pad, pad_phi, pad_psi1, pad_psi2


# Runtime helpers ############################################################
def build_cwt_unpad_indices(N, J_pad, pad_left):
    """`compute_border_indices()` for `cwt()`."""
    padded_len = 2**J_pad

    cwt_unpad_indices = {}
    for hop_size in range(1, N + 1):
        r = padded_len / hop_size
        if r.is_integer():
            n_time = N // hop_size
            ind_start = math.ceil(pad_left / hop_size)
            ind_end = ind_start + n_time
            cwt_unpad_indices[hop_size] = (ind_start, ind_end)
    return cwt_unpad_indices

# Graph builders #############################################################
def build_compute_graph_tm(self):
    """This code was moved from `wavespin.scattering1d.core.scattering1d`
    to avoid repeated compute at runtime and enable certain optimizations. It's
    meant to be read alongside that code.

    It runs each time `oversampling` or `paths_include_n2n1` is updated.

    Commented headers match that of `core`, so some stuff doesn't make
    sense (e.g. "execute").
    """
    # access some attrs directly to avoid recursion
    paths_include_n2n1, psi1_f, psi2_f, log2_T, oversampling = [
        getattr(self, name) for name in
        ('_paths_include_n2n1', 'psi1_f', 'psi2_f', 'log2_T', 'oversampling')]

    # First order ############################################################
    # make compute blocks ####################################################
    U_1_dict = {}
    for n1, p1f in enumerate(psi1_f):
        j1 = p1f['j']
        k1 = max(min(j1, log2_T) - oversampling, 0)
        if k1 not in U_1_dict:
            U_1_dict[k1] = 0
        U_1_dict[k1] += 1

    # execute compute blocks #################################################
    keys1 = []
    offsets = []
    keys1_grouped = {}
    keys1_grouped_inverse = {}
    for n1, p1f in enumerate(psi1_f):
        # Convolution + downsampling
        j1 = p1f['j']
        k1 = max(min(j1, log2_T) - oversampling, 0)

        # Store coefficient in proper grouping
        offset = 0
        for k in U_1_dict:
            if k < k1:
                offset += U_1_dict[k]
        offsets.append(offset)

        keys1.append((k1, n1))
        if k1 not in keys1_grouped:
            keys1_grouped[k1] = []
        keys1_grouped[k1].append(n1)
        keys1_grouped_inverse[n1] = k1

    # Second order ###########################################################
    # make compute blocks ################################################
    U_12_dict = {}
    # here we just append metadata for later use: which n2 will be realized,
    # and their corresponding n1, grouped by k1
    for n2, p2f in enumerate(psi2_f):
        if n2 not in paths_include_n2n1:
            continue
        for n1, (key, p1f) in enumerate(zip(keys1, psi1_f)):
            j1 = p1f['j']
            if n1 not in paths_include_n2n1[n2]:
                continue

            k1, _n1 = key
            assert _n1 == n1, (_n1, n1)

            # for each `n2`,
            if n2 not in U_12_dict:
                U_12_dict[n2] = {}
            # we have `k1`s,
            if k1 not in U_12_dict[n2]:
                U_12_dict[n2][k1] = []
            # with corresponding `n1`s.
            U_12_dict[n2][k1].append(n1)

    # execute compute blocks #############################################
    # used to avoid determining `n1` associated with `S_2` (i.e. fetch it faster)
    n1s_of_n2 = {}
    # used to quickly fetch this quantity as opposed to using `len()`
    n_n1s_for_n2_and_k1 = {}

    for n2 in U_12_dict:
        keys2 = []
        n_n1s_for_n2_and_k1[n2] = {}

        for k1 in U_12_dict[n2]:
            n_n1s_for_n2_and_k1[n2][k1] = len(U_12_dict[n2][k1])
            # Used for sanity check that the right n2-n1 were computed
            keys2.extend(U_12_dict[n2][k1])

        # append into outputs ############################################
        n1s_of_n2[n2] = []
        idx = 0
        for n1, p1f in enumerate(psi1_f):
            if n1 not in paths_include_n2n1[n2]:
                continue
            assert n1 == keys2[idx], (n1, keys2[idx], idx)
            n1s_of_n2[n2].append(n1)
            idx += 1
        # `U_1_dict[k1]` sets `U_1_hats_grouped[k1].shape[1]`. In second order,
        # we fetch `U_1_hats` from it to compute `U_2_hats`, yet not everything
        # in `U_1_hats` can be used to compute `U_2_hats` for a given `n2`,
        # so check we accounted for this.
        assert idx == sum(n_n1s_for_n2_and_k1[n2].values()), (
            idx, n2, n_n1s_for_n2_and_k1)

    # Sanity check: second order #############################################
    # ensure there are no skips in `n1`s, and that for any `n2`, the first
    # `n1` for the same `k1` is the same. This is so that `U_1_hats` is matched
    # correctly, and for theory (energy flow, smart paths)
    _first_n1_for_k1 = {}
    for n2 in U_12_dict:
        for k1 in U_12_dict[n2]:
            n1s = U_12_dict[n2][k1]
            first_n1 = n1s[0]
            # assert "first k1"
            if k1 not in _first_n1_for_k1:
                _first_n1_for_k1[k1] = first_n1
            else:
                assert first_n1 == _first_n1_for_k1[k1], (
                    first_n1, _first_n1_for_k1[k1], k1, n2)
            # assert "noskip n1"
            assert len(n1s) == 1 or np.all(np.diff(n1s) == 1), (n1s, k1, n2)

    # pack & return ##########################################################
    compute_graph = dict(
        U_1_dict=U_1_dict,
        U_12_dict=U_12_dict,
        keys1_grouped=keys1_grouped,
        keys1_grouped_inverse=keys1_grouped_inverse,
        offsets=offsets,
        n1s_of_n2=n1s_of_n2,
        n_n1s_for_n2_and_k1=n_n1s_for_n2_and_k1,
        # unused keys below, kept for debugging
        keys1=keys1,
    )
    return compute_graph

# Meta #######################################################################
def compute_meta_scattering(psi1_f, psi2_f, phi_f, log2_T, paths_include_n2n1,
                            max_order):
    """Get metadata of the Wavelet Time Scattering transform.

    Specifies the content of each scattering coefficient - which order,
    frequencies, filters were used, and so on. See below for more info.

    See `help(wavespin.Scattering1D())` for description of parameters.

    Returns
    -------
    meta : dictionary
        Each value is a tensor, `C` is the total number of scattering coeffs,
        and each tensor is padded with NaNs where appropriate (no valid value).
        Key `'key'` is an exception, which is a list.

        - `'order`' : length `C`
            The total number of scattering coefficients, specifying the
            scattering order.

        - `'xi'` : shape `(C, max_order)`
            The center frequency of the filter used at each order.

        - `'sigma'` : shape `(C, max_order)`
            The frequential bandwidth of the filter used at each order.

        - `'j'` : shape `(C, max_order)`
            The dyadic scale of the filter used at each order.

        - `'is_cqt'` : shape `(C, max_order)`
            Whether the filter was constructed per Constant-Q Transform.

        - `'n'` : shape `(C, max_order)`
            The indices of the filters used at each order.

        - `'key'` : list
            The tuples indexing the corresponding scattering coefficient
            in the non-vectorized output.

        Meta is built such that indexing meta equates indexing coefficients.
        So, `meta['xi'][0]` fetches zeroth-order's `xi`, while `meta['xi'][5]`
        fetches fifth first-order coefficient's `xi`. Each entry is length 2,
        for `(n2, n1)`, i.e. second and first orders, so `meta['xi'][5][1]`
        corresponds to `psi1_f[4]`.

        Note, zeroth-order is packaged along first; this actually isn't an
        inconsistency, `phi_f` is a part of the complete first-order filerbank.

        This docstring wasn't optimized from its original, worth checking
        `help(wavespin.scattering1d.scat_utils_jtfs.compute_meta_jtfs)`.

    References
    ----------
    This is a modification of `kymatio/scattering1d/utils.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    # instantiate
    meta = {}
    for field in ('order', 'xi', 'sigma', 'j', 'is_cqt', 'n', 'key'):
        meta[field] = [[], [], []]

    # Zeroth order
    meta['order'][0].append(0)
    meta['xi'][0].append((phi_f['xi'],))
    meta['sigma'][0].append((phi_f['sigma'],))
    meta['j'][0].append((log2_T,))
    meta['is_cqt'][0].append(())
    meta['n'][0].append(())
    meta['key'][0].append(())

    # First order
    for n1, p1 in enumerate(psi1_f):
        xi1, sigma1, j1, is_cqt1 = [p1[field] for field in
                                    ('xi', 'sigma', 'j', 'is_cqt')]
        meta['order'][1].append(1)
        meta['xi'][1].append((xi1,))
        meta['sigma'][1].append((sigma1,))
        meta['j'][1].append((j1,))
        meta['is_cqt'][1].append((is_cqt1,))
        meta['n'][1].append((n1,))
        meta['key'][1].append((n1,))

    # Second order
    if max_order >= 2:
        for n2, p2 in enumerate(psi2_f):
            if n2 not in paths_include_n2n1:
                continue
            xi2, sigma2, j2, is_cqt2 = [p2[field] for field in
                                        ('xi', 'sigma', 'j', 'is_cqt')]

            for n1, p1 in enumerate(psi1_f):
                if n1 not in paths_include_n2n1[n2]:
                    continue
                xi1, sigma1, j1, is_cqt1 = [p1[field] for field in
                                            ('xi', 'sigma', 'j', 'is_cqt')]
                meta['order'][2].append(2)
                meta['xi'][2].append((xi2, xi1))
                meta['sigma'][2].append((sigma2, sigma1))
                meta['j'][2].append((j2, j1))
                meta['is_cqt'][2].append((is_cqt2, is_cqt1))
                meta['n'][2].append((n2, n1))
                meta['key'][2].append((n2, n1))

    # join orders
    for field, value in meta.items():
        meta[field] = value[0] + value[1] + value[2]

    # left-pad with nans
    pad_fields = ['xi', 'sigma', 'j', 'is_cqt', 'n']
    pad_len = max_order

    for field in pad_fields:
        meta[field] = [(math.nan,) * (pad_len - len(x)) + x
                       for x in meta[field]]

    # to array
    array_fields = ['order', 'xi', 'sigma', 'j', 'is_cqt', 'n']

    for field in array_fields:
        meta[field] = np.array(meta[field])

    return meta
