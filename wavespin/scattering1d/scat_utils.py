# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import numpy as np
import math
import warnings

from .filter_bank import (calibrate_scattering_filters, gauss_1d, morlet_1d,
                          N_and_pad_2_J_pad)
from ..utils.measures import (compute_spatial_support,
                              compute_minimum_required_length)
from ..utils.gen_utils import npy


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
        `2**J_pad` is the support of the padded signal
    N : int
        original signal support size

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
        raise ValueError('Padding support should be larger than the original '
                         'signal size!')
    to_add = 2**J_pad - N
    pad_right = to_add // 2
    pad_left = to_add - pad_right
    return pad_left, pad_right


def compute_minimum_support_to_pad(N, J, Q, T, criterion_amplitude=1e-3,
                                   normalize='l1', r_psi=math.sqrt(0.5),
                                   sigma0=.13, P_max=5, eps=1e-7,
                                   pad_mode='reflect'):
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
        Supports `'l1'`, `'l2'`, `'l1-energy'`, `'l2-energy'`, but only
        `'l1'` or `'l2'` is used. See `help(Scattering1D)`.

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

    pad_mode : str
        Name of padding used. If `'zero'`, will halve `min_to_pad`, else no
        effect.

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

    # can pad half as much
    if pad_mode == 'zero':
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


def build_compute_graph_tm(self):
    """This code was moved from `wavespin.scattering1d.core.scattering1d`
    to avoid repeated compute at runtime.

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


def build_compute_graph_fr(self):  # TODO scat_utils_jtfs
    # unpack some attributes #################################################
    (scf, paths_include_build, paths_exclude, psi2_f, log2_T, average_global_phi,
     oversampling, oversampling_fr) = [
         getattr(self, k) for k in
         ('scf', 'paths_include_build', 'paths_exclude', 'psi2_f', 'log2_T',
          'average_global_phi', 'oversampling', 'oversampling_fr')
    ]
    # Time scattering, early pairs ###########################################
    Dearly = _compute_graph_fr_tm(self)

    # Joint complex compute graph ############################################
    # plan compute graph
    Y_1_fr_dict = {}
    for scale_diff in scf.scale_diffs_unique:
        # make `scale_diff`-dependent since `n1_fr_subsample` is
        Y_1_fr_dict[scale_diff] = {}
        n1_fr_subsamples = scf.n1_fr_subsamples['spinned'][scale_diff]

        # group by `n1_fr_subsample`
        psi_id = scf.psi_ids[scale_diff]
        for n1_fr in range(len(scf.psi1_f_fr_up[psi_id])):
            if n1_fr in paths_exclude['n1_fr']:
                continue
            n1_fr_subsample = max(n1_fr_subsamples[n1_fr] - oversampling_fr, 0)
            if n1_fr_subsample not in Y_1_fr_dict[scale_diff]:
                Y_1_fr_dict[scale_diff][n1_fr_subsample] = []
            Y_1_fr_dict[scale_diff][n1_fr_subsample].append(n1_fr)

    # `U1 * (psi_t * psi_f)`, `U1 * (psi_t * phi_f)` #########################
    DT, DF, DL = {}, {}, {}

    for n2 in paths_include_build:
        # `g:` for "group:"
        DT[n2] = {}
        DF[n2] = {'g:n1_frs': {}, 'g:n1_fr_subsamples': {}}
        DL[n2] = {'g:n1_frs': {}, 'g:n1_fr_subsamples': {}}

        j2 = psi2_f[n2]['j']

        # `k1_plus_k2` is same for all `k1 + k2` if they were computed the
        # long way (i.e. as in Scattering1D)
        k1_plus_k2 = max(min(j2, log2_T) - oversampling, 0)
        maybe_unpad_time_info = _compute_graph_maybe_unpad_time(k1_plus_k2, self)

        # frequential pad
        scale_diff = scf.scale_diffs[n2]
        pad_fr = scf.J_pad_frs[scale_diff]

        # pack ---------------------------------------------------------------
        DT[n2].update(dict(
            j2=j2,
            k1_plus_k2=k1_plus_k2,
            pad_fr=pad_fr,
            **maybe_unpad_time_info,
        ))

        # `_frequency_scattering()`, `_joint_lowpass()` ----------------------
        _compute_graph_frequency_scattering(n2, pad_fr, DT, DF, DL, self)

        # `_frequency_lowpass()`, `_joint_lowpass()` -------------------------
        _compute_graph_frequency_lowpass(n2, pad_fr, DT, DF, DL, self)

    # `U1 * (phi_t * psi_f)` #################################################
    # take largest subsampling factor
    n2 = -1
    j2 = log2_T
    k1_plus_k2 = (max(log2_T - oversampling, 0) if not average_global_phi else
                  log2_T)
    pad_fr = scf.J_pad_frs_max
    # n2_time = U_0.shape[-1] // 2**max(j2 - oversampling, 0)
    trim_tm = 0

    # pack -------------------------------------------------------------------
    DT[n2] = {}
    DF[n2] = {'g:n1_frs': {}, 'g:n1_fr_subsamples': {}}
    DL[n2] = {'g:n1_frs': {}, 'g:n1_fr_subsamples': {}}

    DT[n2].update(dict(
        j2=j2,
        k1_plus_k2=k1_plus_k2,
        trim_tm=trim_tm,
        pad_fr=pad_fr,
    ))
    _compute_graph_frequency_scattering(n2, pad_fr, DT, DF, DL, self)

    # Determine `n1_fr`-dependence of `_joint_lowpass_part_2()` ##############
    for n2 in DF:
        part2_grouped_by_n1_fr_subsample = True  # TODO part_2
        n1_fr_subsample_prev = None
        for n1_fr in DF[n2]['g:n1_frs']:
            if n1_fr == -1:
                # grouping only concerns spinned
                continue
            elif n1_fr in paths_exclude['n1_fr']:
                continue

            n1_fr_subsample = DF[n2]['g:n1_frs'][n1_fr]['n1_fr_subsample']
            # for new `n1_fr_subsample`, refresh `params_prev`
            if n1_fr_subsample != n1_fr_subsample_prev:
                params_prev = None

            # DL disagreements should follow DF disagreements, but check both
            # to be safe
            params_now = (DF[n2]['g:n1_frs'][n1_fr],
                          DL[n2]['g:n1_frs'][n1_fr])
            # if `params_prev` exists and disagrees with `params_now`, can't group
            if params_prev is not None and params_now != params_prev:
                part2_grouped_by_n1_fr_subsample = False
                break
            elif params_prev is None:
                params_prev = params_now
            n1_fr_subsample_prev = n1_fr_subsample

        if not part2_grouped_by_n1_fr_subsample:
            # shouldn't be possible otherwise
            assert oversampling_fr > 0 or (scf.average_fr and not scf.aligned)
        DF[n2]['part2_grouped_by_n1_fr_subsample'
               ] = part2_grouped_by_n1_fr_subsample

    # for those that can be grouped, group them
    for n2 in DF:
        if DF[n2]['part2_grouped_by_n1_fr_subsample']:
            for n1_fr in DF[n2]['g:n1_frs']:
                if n1_fr == -1:  # doesn't need grouping
                    continue
                n1_fr_subsample = DF[n2]['g:n1_frs'][n1_fr]['n1_fr_subsample']
                DF[n2]['g:n1_fr_subsamples'][n1_fr_subsample
                                             ].update(DF[n2]['g:n1_frs'][n1_fr])

    # pack & return ##########################################################
    compute_graph = dict(Dearly=Dearly, Y_1_fr_dict=Y_1_fr_dict,
                         DT=DT, DF=DF, DL=DL)
    return compute_graph


def _compute_graph_frequency_scattering(n2, pad_fr, DT, DF, DL, self):
    scf, paths_exclude, average_fr, oversampling_fr = [
        getattr(self, k) for k in
        ('scf', 'paths_exclude', 'average_fr', 'oversampling_fr')
    ]

    scale_diff = scf.scale_diffs[n2]
    psi_id = scf.psi_ids[scale_diff]
    pad_diff = scf.J_pad_frs_max_init - pad_fr

    n1_fr_subsamples = scf.n1_fr_subsamples['spinned'][scale_diff]
    log2_F_phi_diffs = scf.log2_F_phi_diffs['spinned'][scale_diff]

    # unpad early if possible
    # (note, incomplete criterion - we could do it like the "maybe unpad time"
    # method, but not often - avoid complicating)
    unpad_early_fr = bool(not average_fr)

    # pack n2-only dependents
    DF[n2].update(dict(
        psi_id=psi_id,
        pad_diff=pad_diff,
        n1_fr_subsamples=n1_fr_subsamples,
        log2_F_phi_diffs=log2_F_phi_diffs,
        unpad_early_fr=unpad_early_fr,
    ))

    # core logic
    for n1_fr in range(len(scf.psi1_f_fr_up[psi_id])):
        if n1_fr in paths_exclude['n1_fr']:
            continue
        # compute subsampling
        total_conv_stride_over_U1 = scf.total_conv_stride_over_U1s[
            scale_diff][n1_fr]
        log2_F_phi_diff = log2_F_phi_diffs[n1_fr]
        # more like `n1_fr_subsample_realized`
        n1_fr_subsample = max(n1_fr_subsamples[n1_fr] - oversampling_fr, 0)

        # pack
        DF[n2]['g:n1_frs'][n1_fr] = dict(
            total_conv_stride_over_U1=total_conv_stride_over_U1,
            log2_F_phi_diff=log2_F_phi_diff,
            n1_fr_subsample=n1_fr_subsample,
        )
        if n1_fr_subsample not in DF[n2]['g:n1_fr_subsamples']:
            DF[n2]['g:n1_fr_subsamples'][n1_fr_subsample] = {}  # for later

        # `_joint_lowpass()` -------------------------------------------------
        _compute_graph_joint_lowpass(n2, n1_fr, DT, DF, DL, self)

    # compute `n_n1s` based on `n1_fr_subsample`
    for n1_fr_subsample in DF[n2]['g:n1_fr_subsamples']:
        _D = DL[n2]['g:n1_fr_subsamples'][n1_fr_subsample]
        if unpad_early_fr:
            n_n1s = _D['ind_end_fr'] - _D['ind_start_fr']
        else:
            n_n1s = 2**(pad_fr - n1_fr_subsample)
        DF[n2]['g:n1_fr_subsamples'][n1_fr_subsample][
            'n_n1s_pre_part_2'] = n_n1s


def _compute_graph_joint_lowpass(n2, n1_fr, DT, DF, DL, self):
    # unpack #################################################################
    pad_fr, k1_plus_k2, trim_tm = [
        DT[n2][k] for k in
        ('pad_fr', 'k1_plus_k2', 'trim_tm')
    ]
    (total_conv_stride_over_U1, n1_fr_subsample, log2_F_phi_diff) = [
        DF[n2]['g:n1_frs'][n1_fr][k] for k in
        ('total_conv_stride_over_U1', 'n1_fr_subsample', 'log2_F_phi_diff')
    ]
    (scf, ind_start, ind_end, average, average_global, average_fr,
     out_3D, oversampling, oversampling_fr, log2_T, N,
     do_energy_correction) = [
         getattr(self, k) for k in
         ('scf', 'ind_start', 'ind_end', 'average', 'average_global', 'average_fr',
          'out_3D', 'oversampling', 'oversampling_fr', 'log2_T', 'N',
          'do_energy_correction')
    ]

    # compute subsampling logic ##############################################
    global_averaged_fr = (scf.average_fr_global if n1_fr != -1 else
                          scf.average_fr_global_phi)
    if global_averaged_fr:
        lowpass_subsample_fr = total_conv_stride_over_U1 - n1_fr_subsample
    elif scf.average_fr:
        lowpass_subsample_fr = max(total_conv_stride_over_U1 - n1_fr_subsample
                                   - scf.oversampling_fr, 0)
    else:
        lowpass_subsample_fr = 0

    # compute freq unpad params ##############################################
    total_conv_stride_over_U1_realized = n1_fr_subsample + lowpass_subsample_fr
    do_averaging    = bool(average    and n2    != -1)
    do_averaging_fr = bool(average_fr and n1_fr != -1)

    # "conventional" unpadding with respect to N_fr
    ind_start_N_fr = scf.ind_start_fr[n2][total_conv_stride_over_U1_realized]
    ind_end_N_fr   = scf.ind_end_fr[  n2][total_conv_stride_over_U1_realized]
    if scf.out_3D:
        # Unpad length must be same for all `(n2, n1_fr)`; this is determined by
        # the longest N_fr, hence we compute the reference quantities there.
        _stride_ref = scf.total_conv_stride_over_U1s[0][0]
        stride_ref = max(_stride_ref - scf.oversampling_fr, 0)
        ind_start_fr = scf.ind_start_fr_max[stride_ref]
        ind_end_fr   = scf.ind_end_fr_max[  stride_ref]
    else:
        ind_start_fr, ind_end_fr = ind_start_N_fr, ind_end_N_fr

    # unpad early if possible ################################################
    # here we'd handle early unpadding along time and frequency, but
    # in current implementation both happen before this point
    unpadded_tm, unpadded_fr = (not do_averaging), (not do_averaging_fr)

    # time lowpassing ########################################################
    if do_averaging:
        if average_global:
            total_conv_stride_tm = log2_T
        elif average:
            # Low-pass filtering over time
            k2_log2_T = max(log2_T - k1_plus_k2 - oversampling, 0)
            total_conv_stride_tm = k1_plus_k2 + k2_log2_T
    else:
        total_conv_stride_tm = k1_plus_k2
    ind_start_tm = ind_start[trim_tm][total_conv_stride_tm]
    ind_end_tm   = ind_end[  trim_tm][total_conv_stride_tm]

    # `not average` and `n2 == -1` already unpadded
    do_unpad_tm = bool(not average_global and not unpadded_tm)

    # freq lowpassing ########################################################
    if do_averaging_fr:
        pass
    else:
        pass

    # unpad only if input isn't global averaged
    do_unpad_fr = bool(not global_averaged_fr and not unpadded_fr)
    # repad only if input isn't global averaged
    do_repad_fr = bool(not global_averaged_fr and
                       2**pad_fr < ind_end_fr - ind_start_fr)
    # fr repad should only occur in the following case
    if do_repad_fr:
        assert out_3D and oversampling_fr > 0

    # energy correction ######################################################
    if do_energy_correction:
        do_ec_frac_tm, do_ec_frac_fr = _get_do_ec_frac(
            self, tm=True, fr=True, pad_fr=pad_fr)
        param_tm = (do_ec_frac_tm, N, ind_start_tm, ind_end_tm,
                    total_conv_stride_tm)
        param_fr = (do_ec_frac_fr, scf.N_frs[n2], ind_start_fr, ind_end_fr,
                    total_conv_stride_over_U1_realized)

        if n2 != -1:
            # `n2=-1` already did time
            _kw = dict(param_tm=param_tm, param_fr=param_fr, phi_t_psi_f=False)
        else:
            _kw = dict(param_fr=param_fr, phi_t_psi_f=True)
        energy_correction = _compute_energy_correction_factor(**_kw)

    # pack ###################################################################
    info = dict(
        global_averaged_fr=global_averaged_fr,
        lowpass_subsample_fr=lowpass_subsample_fr,
        do_averaging=do_averaging,
        do_averaging_fr=do_averaging_fr,
        total_conv_stride_over_U1_realized=total_conv_stride_over_U1_realized,
        ind_start_fr=ind_start_fr,
        ind_end_fr=ind_end_fr,
        ind_start_N_fr=ind_start_N_fr,
        ind_end_N_fr=ind_end_N_fr,
        unpadded_tm=unpadded_tm,
        unpadded_fr=unpadded_fr,
        do_unpad_tm=do_unpad_tm,
        do_unpad_fr=do_unpad_fr,
        do_repad_fr=do_repad_fr,
        total_conv_stride_tm=total_conv_stride_tm,
        ind_start_tm=ind_start_tm,
        ind_end_tm=ind_end_tm,
    )
    if do_energy_correction:
        info.update(dict(
            energy_correction=energy_correction,
            param_tm=param_tm,
            param_fr=param_fr,
        ))
    if do_averaging and not average_global:
        info['k2_log2_T'] = k2_log2_T
    if out_3D:
        info['stride_ref'] = stride_ref
    DL[n2]['g:n1_frs'][n1_fr] = info
    # `n1_fr = -1` case doesn't need grouping
    if n1_fr != -1:
        if n1_fr_subsample not in DL[n2]['g:n1_fr_subsamples']:
            DL[n2]['g:n1_fr_subsamples'][n1_fr_subsample] = info.copy()
        else:
            # If a key-value pair doesn't match an existing value, delete the key.
            # This way, we only group keys which are groupable by
            # `n1_fr_subsample`.
            # Note however, just because a key wasn't deleted, doesn't mean it
            # lacks `n1_fr`-dependence in all variants of this configuration
            # (could be luck).
            D = DL[n2]['g:n1_fr_subsamples'][n1_fr_subsample]
            for k, v in info.items():
                if k in D and v != D[k]:
                    del D[k]


def _compute_graph_frequency_lowpass(n2, pad_fr, DT, DF, DL, self):
    scf = self.scf

    if scf.average_fr_global_phi:
        # `min` in case `pad_fr > N_fr_scales_max` or `log2_F > pad_fr`
        total_conv_stride_over_U1_phi = min(pad_fr, scf.log2_F)
        n1_fr_subsample = total_conv_stride_over_U1_phi
        log2_F_phi = scf.log2_F
        log2_F_phi_diff = scf.log2_F - log2_F_phi
    else:
        # fetch stride params
        scale_diff = scf.scale_diffs[n2]
        total_conv_stride_over_U1_phi = scf.total_conv_stride_over_U1s_phi[
            scale_diff]
        n1_fr_subsample = max(scf.n1_fr_subsamples['phi'][scale_diff] -
                              scf.oversampling_fr, 0)
        # scale params
        log2_F_phi = scf.log2_F_phis['phi'][scale_diff]
        log2_F_phi_diff = scf.log2_F_phi_diffs['phi'][scale_diff]

    # "early" in that it precedes `_joint_lowpass()`
    unpad_early_fr = bool(not scf.average_fr_global_phi)

    DF[n2]['g:n1_frs'][-1] = dict(
        total_conv_stride_over_U1=total_conv_stride_over_U1_phi,
        log2_F_phi=log2_F_phi,
        log2_F_phi_diff=log2_F_phi_diff,
        n1_fr_subsample=n1_fr_subsample,
        unpad_early_fr=unpad_early_fr,
    )

    _compute_graph_joint_lowpass(n2, -1, DT, DF, DL, self)


def _compute_graph_maybe_unpad_time(k1_plus_k2, self):
    # unpack #################################################################
    (phi_f, pad_mode, ind_start, ind_end, average,
     J, J_pad, log2_T, N) = [
        getattr(self, k) for k in
        ('phi_f', 'pad_mode', 'ind_start', 'ind_end', 'average',
         'J', 'J_pad', 'log2_T', 'N')
    ]
    # ------------------------------------------------------------------------
    do_unpad_dyadic, do_unpad = False, False
    info = {}

    start, end = ind_start[0][k1_plus_k2], ind_end[0][k1_plus_k2]
    if average and log2_T < J[0]:
        # compute padding currently needed for lowpass filtering
        min_to_pad = phi_f['support'][0]
        if pad_mode == 'zero':
            min_to_pad //= 2
        pad_log2_T = N_and_pad_2_J_pad(N, min_to_pad) - k1_plus_k2

        # compute current dyadic length; `2**padded_current == Y_2_c.shape[-1]`
        padded_current = J_pad - k1_plus_k2

        # compute permitted unpadding amounts, which impose different restrictions
        N_scale = math.ceil(math.log2(N))
        # need `max` in case `max_pad_factor` makes `J_pad < pad_log2_T`
        # (and thus `padded < pad_log2_T`); need >=0 for `trim_tm`, which
        # is needed for indexing later
        unpad_allowed_for_lowpassing = int(max(0, padded_current - pad_log2_T))
        max_unpad_that_counts_as_unpad = J_pad - N_scale
        amount_can_unpad_now = min(unpad_allowed_for_lowpassing,
                                   max_unpad_that_counts_as_unpad)
        if amount_can_unpad_now > 0:
            do_unpad_dyadic = True
            info['unpad_dyadic_kwargs'] = {
                'N': end - start,
                'orig_padded_len': 2**J_pad,
                'target_padded_len': 2**pad_log2_T,
                'k': k1_plus_k2
            }
        trim_tm = amount_can_unpad_now
    elif not average:
        # can unpad fully, no further processing along time
        do_unpad = True
        info['unpad_kwargs'] = {'i0': start, 'i1': end}
        # no longer applicable, but still needed for computing final time length
        # (i.e. won't use it to fetch `phi_f`, but will use it to fetch unpad
        # indices that'll match the early unpadding we'll do)
        trim_tm = 0
    else:
        # no unpadding, use full length of `phi_f`
        trim_tm = 0

    # pack & return
    info.update(dict(trim_tm=trim_tm,
                     do_unpad=do_unpad,
                     do_unpad_dyadic=do_unpad_dyadic))
    return info


def _compute_graph_fr_tm(self):
    """Computes quantities for handling pairs that don't utilize
    `_joint_lowpass()`:

        S0, S1, phi_t * phi_f, `phi_t *` portion
    """
    (scf, N, average, average_global, average_global_phi, log2_T,
     oversampling, oversampling_fr, ind_start, ind_end, vectorized,
     out_3D, out_exclude, do_energy_correction) = [
         getattr(self, k) for k in
         ('scf', 'N', 'average', 'average_global', 'average_global_phi', 'log2_T',
          'oversampling', 'oversampling_fr', 'ind_start', 'ind_end', 'vectorized',
          'out_3D', 'out_exclude', 'do_energy_correction')
    ]
    if out_exclude is None:
        out_exclude = []
    if do_energy_correction:
        # get fractional unpad energy correction info
        do_ec_frac_tm = _get_do_ec_frac(self, tm=True)

    Dearly = {}

    # Zeroth order ###########################################################
    if 'S0' not in out_exclude:
        # compute `k0`
        if average_global:
            k0 = log2_T
        elif average:
            k0 = max(log2_T - oversampling, 0)
        # fetch unpad indices
        if average:
            ind_start_tm, ind_end_tm = ind_start[0][k0], ind_end[0][k0]
        else:
            ind_start_tm, ind_end_tm = ind_start[0][0], ind_end[0][0]

        if do_energy_correction and average:
            S0_ec = _compute_energy_correction_factor(
                param_tm=(do_ec_frac_tm, N, ind_start_tm, ind_end_tm, k0),
            )

        # pack
        Dearly['S0'] = dict(
            ind_start_tm=ind_start_tm,
            ind_end_tm=ind_end_tm,
        )
        if average:
            Dearly['S0']['k0'] = k0
            if do_energy_correction:
                Dearly['S0']['energy_correction'] = S0_ec

    # Time scattering ########################################################
    jtfs_cfg_in_s1d = {}
    # whether `phi_t *` pairs are needed
    include_phi_t = any(pair not in out_exclude for pair in
                        ('phi_t * phi_f', 'phi_t * psi_f'))
    # if both are false then `S_1_avg` is never used, regardless of `average`
    need_S_1_avg = bool(include_phi_t or 'S1' not in out_exclude)
    # whether to compute `S1` as if we're just using `Scattering1D` (only its
    # arguments)
    jtfs_cfg_in_s1d['do_S_1_tm'] = bool('S1' not in out_exclude)
    # whether to compute `S1` with forced `average=True`
    jtfs_cfg_in_s1d['do_S_1_avg'] = bool((average or include_phi_t) and
                                         need_S_1_avg)
    # same for `S0`
    jtfs_cfg_in_s1d['do_S_0'] = bool('S0' not in out_exclude)
    # whether to skip spinned
    skip_spinned = bool('psi_t * psi_f_up' in out_exclude and
                        'psi_t * psi_f_dn' in out_exclude)
    # whether to do joint scattering that requires second-order time scattering
    do_joint_complex = bool(not (skip_spinned and
                                 'psi_t * phi_f' in out_exclude))
    jtfs_cfg_in_s1d['jtfs_needs_U_2_c'] = bool(do_joint_complex)
    # pack this since time scattering doesn't use it
    jtfs_cfg_in_s1d['average_global_phi'] = average_global_phi

    # pack
    Dearly.update(
        jtfs_cfg_in_s1d=jtfs_cfg_in_s1d,
        include_phi_t=include_phi_t,
        do_joint_complex=do_joint_complex,
        skip_spinned=skip_spinned,
    )

    # `S1` or `phi_t *` ######################################################
    U_1_dict, U_12_dict, keys1_grouped_inverse = [
        self.compute_graph[k] for k in
        ('U_1_dict', 'U_12_dict', 'keys1_grouped_inverse')
    ]

    total_conv_stride_tms, total_conv_stride_tm_avg = {}, None
    ind_start_tms, ind_start_tm_avg = {}, None
    ind_end_tms, ind_end_tm_avg = {}, None

    for k1 in U_1_dict:
        # subsampling factors
        k1_log2_T = max(log2_T - k1 - oversampling, 0)
        # stride & unpad indices for `phi_t *`
        if average or include_phi_t:
            if not average_global_phi:
                _total_conv_stride_tm_avg = k1 + k1_log2_T
            else:
                _total_conv_stride_tm_avg = log2_T
            _ind_start_tm_avg = ind_start[0][_total_conv_stride_tm_avg]
            _ind_end_tm_avg = ind_end[0][_total_conv_stride_tm_avg]
        # stride & unpad indices for `S1`
        total_conv_stride_tm = (_total_conv_stride_tm_avg if average else
                                k1)
        ind_start_tm = ind_start[0][total_conv_stride_tm]
        ind_end_tm = ind_end[0][total_conv_stride_tm]

        # store
        total_conv_stride_tms[k1] = total_conv_stride_tm
        ind_start_tms[k1] = ind_start_tm
        ind_end_tms[k1] = ind_end_tm
        if average or include_phi_t:
            if total_conv_stride_tm_avg is None:
                total_conv_stride_tm_avg = _total_conv_stride_tm_avg
                ind_start_tm_avg = _ind_start_tm_avg
                ind_end_tm_avg = _ind_end_tm_avg
            else:
                # should be same for all `k1`
                assert _total_conv_stride_tm_avg == total_conv_stride_tm_avg
                assert _ind_start_tm_avg == ind_start_tm_avg
                assert _ind_end_tm_avg == ind_end_tm_avg

    # S1: execute
    if 'S1' not in out_exclude:
        # energy correction due to stride & inexact unpad length
        if do_energy_correction:
            if vectorized and average:
                S1_ec = _compute_energy_correction_factor(
                    param_tm=(do_ec_frac_tm, N,
                              ind_start_tms[0], ind_end_tms[0],
                              total_conv_stride_tms[0]),
                )
            else:
                S1_ec = {}
                for n1, k1 in keys1_grouped_inverse.items():
                    S1_ec[k1] = _compute_energy_correction_factor(
                        param_tm=(do_ec_frac_tm, N,
                                  ind_start_tms[k1], ind_end_tms[k1],
                                  total_conv_stride_tms[k1]),
                    )

    # `phi_t *`: append for further processing
    if include_phi_t:
        # energy correction, if not already done
        if do_energy_correction:
            phi_t_ec = _compute_energy_correction_factor(
                param_tm=(do_ec_frac_tm, N, ind_start_tm_avg, ind_end_tm_avg,
                          total_conv_stride_tm_avg),
            )
    # ------------------------------------------------------------------------
    # pack
    Dearly['S1'] = dict(
        ind_start_tms=ind_start_tms,
        ind_end_tms=ind_end_tms,
        total_conv_stride_tms=total_conv_stride_tms,
    )
    if average or include_phi_t:
        Dearly['S1'].update(dict(
            ind_start_tm_avg=ind_start_tm_avg,
            ind_end_tm_avg=ind_end_tm_avg,
            total_conv_stride_tm_avg=total_conv_stride_tm_avg,
        ))
    if do_energy_correction:
        if 'S1' not in out_exclude:
            Dearly['S1']['energy_correction'] = S1_ec
        if include_phi_t:
            Dearly['phi_t']= dict(energy_correction=phi_t_ec)

    # `phi_t * phi_f` ########################################################
    if 'phi_t * phi_f' not in out_exclude:
        pad_fr = scf.J_pad_frs_max
        lowpass_subsample_fr = 0  # no lowpass after lowpass

        if scf.average_fr_global_phi:
            n1_fr_subsample = scf.log2_F
            log2_F_phi = scf.log2_F
        else:
            # this is usually 0
            pad_diff = scf.J_pad_frs_max_init - pad_fr

            # ensure stride is zero if `not average and aligned`
            scale_diff = 0
            log2_F_phi = scf.log2_F_phis['phi'][scale_diff]
            log2_F_phi_diff = scf.log2_F_phi_diffs['phi'][scale_diff]
            n1_fr_subsample = max(scf.n1_fr_subsamples['phi'][scale_diff] -
                                  oversampling_fr, 0)

        # compute for unpad & energy correction
        _stride = n1_fr_subsample + lowpass_subsample_fr
        if out_3D:
            ind_start_fr = scf.ind_start_fr_max[_stride]
            ind_end_fr   = scf.ind_end_fr_max[  _stride]
        else:
            ind_start_fr = scf.ind_start_fr[-1][_stride]
            ind_end_fr   = scf.ind_end_fr[-1][  _stride]

        # set reference for later
        total_conv_stride_over_U1_realized = (n1_fr_subsample +
                                              lowpass_subsample_fr)

        # energy correction due to stride & inexact unpad indices
        # time already done
        if do_energy_correction:
            do_ec_frac_fr = _get_do_ec_frac(self, fr=True, pad_fr=pad_fr)
            phi_t_phi_f_ec = _compute_energy_correction_factor(
                param_fr=(do_ec_frac_fr, scf.N_frs_max, ind_start_fr, ind_end_fr,
                          total_conv_stride_over_U1_realized),
            )

        stride = (total_conv_stride_over_U1_realized, total_conv_stride_tm_avg)

        # ---------------------------------------------------------------------
        # pack
        Dearly['phi_t * phi_f'] = dict(
            lowpass_subsample_fr=lowpass_subsample_fr,
            n1_fr_subsample=n1_fr_subsample,
            log2_F_phi=log2_F_phi,
            stride=stride,
            ind_start_fr=ind_start_fr,
            ind_end_fr=ind_end_fr,
            total_conv_stride_over_U1_realized=total_conv_stride_over_U1_realized,
            total_conv_stride_tm_avg=total_conv_stride_tm_avg,
        )
        if do_energy_correction:
            Dearly['phi_t * phi_f']['energy_correction'] = phi_t_phi_f_ec
        if not scf.average_fr_global_phi:
            Dearly['phi_t * phi_f'].update(dict(
                log2_F_phi_diff=log2_F_phi_diff,
                pad_diff=pad_diff,
            ))

    # return
    return Dearly


def make_psi1_f_fr_stacked_dict(scf, paths_exclude):
    oversampling_fr = scf.oversampling_fr
    # first clear the existing attribute, for memory
    scf.psi1_f_fr_stacked_dict = {}

    psi1_f_fr_stacked_dict = {}
    for scale_diff in scf.scale_diffs_unique:
        psi_id = scf.psi_ids[scale_diff]
        if psi_id in psi1_f_fr_stacked_dict:
            continue
        psi1_f_fr_stacked_dict[psi_id] = {}
        n1_fr_subsamples = scf.n1_fr_subsamples['spinned'][scale_diff]

        if scf.vectorized_early_fr:
            # pack all together
            if not paths_exclude['n1_fr']:
                # this is just here for readability
                psi1_f_fr_stacked_dict[psi_id] = (scf.psi1_f_fr_up[psi_id],
                                                  scf.psi1_f_fr_dn[psi_id])
            else:
                ups, dns = [
                    [pf for n1_fr, pf in enumerate(psi1_f_frs[psi_id])
                     if n1_fr not in paths_exclude['n1_fr']]
                    for psi1_f_frs in (scf.psi1_f_fr_up, scf.psi1_f_fr_dn)
                ]
                psi1_f_fr_stacked_dict[psi_id] = (ups, dns)
        else:
            # group by `n1_fr_subsample`
            for n1_fr in range(len(scf.psi1_f_fr_up[psi_id])):
                if n1_fr in paths_exclude['n1_fr']:
                    continue
                n1_fr_subsample = max(n1_fr_subsamples[n1_fr] -
                                      oversampling_fr, 0)
                # append for stacking
                if n1_fr_subsample not in psi1_f_fr_stacked_dict[psi_id]:
                    psi1_f_fr_stacked_dict[psi_id][n1_fr_subsample] = ([], [])
                psi1_f_fr_stacked_dict[psi_id][n1_fr_subsample][0].append(
                    scf.psi1_f_fr_up[psi_id][n1_fr])
                psi1_f_fr_stacked_dict[psi_id][n1_fr_subsample][1].append(
                    scf.psi1_f_fr_dn[psi_id][n1_fr])

    # do stacking
    for psi_id in psi1_f_fr_stacked_dict:
        # broadcast along batch dim
        if scf.vectorized_early_fr:
            psi1_f_fr_stacked_dict[psi_id] = np.stack(
                npy(psi1_f_fr_stacked_dict[psi_id]))[None]
        else:
            for n1_fr_subsample in psi1_f_fr_stacked_dict[psi_id]:
                psi1_f_fr_stacked_dict[psi_id][n1_fr_subsample] = np.stack(
                    npy(psi1_f_fr_stacked_dict[psi_id][n1_fr_subsample]))[None]

    return psi1_f_fr_stacked_dict


def pack_runtime_spinned(scf, compute_graph_fr, out_exclude):
    """Also validates `psi1_f_fr_stacked_dict` against `Y_1_fr_dict`."""
    if out_exclude is None:
        out_exclude = []

    # Validate stacked filters against compute graph -------------------------
    # this only checks that there's the right number of filters
    if scf.vectorized_fr:
        stacked = scf.psi1_f_fr_stacked_dict  # shorthand
        Y_1_fr_dict = compute_graph_fr['Y_1_fr_dict']
        for scale_diff in Y_1_fr_dict:
            psi_id = scf.psi_ids[scale_diff]
            if scf.vectorized_early_fr:
                assert (stacked[psi_id].shape[2] ==
                        sum(map(len, Y_1_fr_dict[scale_diff].values())))
            else:
                for n1_fr_subsample in Y_1_fr_dict[scale_diff]:
                    assert (stacked[psi_id][n1_fr_subsample].shape[2] ==
                            len(Y_1_fr_dict[scale_diff][n1_fr_subsample]))

    # Determine the spins to be computed -------------------------------------
    spin_data = {}
    psi_ids = np.unique(list(scf.psi_ids.values()))
    for spin_down in (True, False):
        spin_data[spin_down] = {}
        spins = []
        # SPIN NOTE: to compute up, we convolve with *down*, since `U1` is
        # time-reversed relative to the sampling of the wavelets.
        if ('psi_t * psi_f_up' not in out_exclude or
                'phi_t * psi_f' not in out_exclude):
            spins.append(1 if spin_down else 0)
        if spin_down and 'psi_t * psi_f_dn' not in out_exclude:
            spins.append(-1)

        for psi_id in psi_ids:
            if not scf.vectorized_fr:
                psi1_f_frs = []
                if 1 in spins or 0 in spins:
                    psi1_f_frs.append(scf.psi1_f_fr_dn[psi_id])
                if -1 in spins:
                    psi1_f_frs.append(scf.psi1_f_fr_up[psi_id])
            else:
                if len(spins) == 2:
                    if scf.vectorized_early_fr:
                        psi1_f_fr_stacked = scf.psi1_f_fr_stacked_dict[
                            psi_id]
                    else:
                        psi1_f_fr_stacked_subdict = scf.psi1_f_fr_stacked_dict[
                            psi_id]
                else:
                    s_idx = (1 if (1 in spins or 0 in spins) else
                             0)
                    if scf.vectorized_early_fr:
                        psi1_f_fr_stacked = scf.psi1_f_fr_stacked_dict[
                            psi_id][:, s_idx:s_idx + 1]
                    else:
                        psi1_f_fr_stacked_subdict = {}
                        for n1_fr_subsample in scf.psi1_f_fr_stacked_dict[psi_id]:
                            psi1_f_fr_stacked_subdict[n1_fr_subsample] = (
                                scf.psi1_f_fr_stacked_dict[
                                    psi_id][n1_fr_subsample][:, s_idx:s_idx + 1]
                            )

            # pack
            if scf.vectorized_fr:
                if scf.vectorized_early_fr:
                    d = (psi1_f_fr_stacked, spins)
                else:
                    d = (psi1_f_fr_stacked_subdict, spins)
            else:
                d = (psi1_f_frs, spins)
            spin_data[spin_down][psi_id] = d
    return spin_data


def _get_do_ec_frac(self, tm=False, fr=False, pad_fr=None):
    """See `_compute_energy_correction_factor`."""
    ecs = []
    if tm:
        if self.do_ec_frac_tm is None:
            ecs.append(bool(self.pad_mode != 'zero'))
        else:
            ecs.append(bool(self.do_ec_frac_tm))
    if fr:
        if self.do_ec_frac_fr is None:
            ecs.append(bool(self.pad_mode_fr != 'zero' and
                            self.scf.average_fr_global and
                            pad_fr > self.scf.N_fr_scales_max))
        else:
            ecs.append(bool(self.do_ec_frac_fr))
    return tuple(ecs)


def _compute_energy_correction_factor(param_tm=None, param_fr=None,
                                      phi_t_psi_f=False):
    """Enables `||jtfs(x)|| = ||x||`, and `||jtfs_subsampled(x)|| = ||jtfs(x)||`.

    Overview
    --------

    **Stride**:

        Unaliased subsampling by 2 divides energy by 2.
        Make up for it with `coef *= sqrt(2)`.

    **Unpadding**:

        Suppose input length is 64, and stride is 8. Then, `len(out) = 8`.

        Suppose input length is 65, and stride is 8. Then, `len(out) = 9`,
        yet the exact length is `65/8 = 8.125`, which is unachievable. Hence,
        because of just one sample, we have an up to `9/8` energy difference
        relative to the first case - or, to be exact, `9/8.125`.
        Make up for it with `coef *= sqrt(8.125/9)`.

        Note, above is only a partial solution, since unpadding aliases and
        is lossy with respect to padded, and only padded's energy is conserved.
        Since stride and unpadding commute, unpadding strided is same as striding
        unpadded, which can be aliased even while striding padded isn't.

    Motivation
    ----------
    Equalizing expected norm across sampling rates, akin to "inverted dropout"
    in neural networks:

        E(||Sx_subsampled||^2) ~= E(||Sx||^2)

    where `E` is the expectation operator, and `||x||` is the Euclidean/L2 norm.
    In particular we wish to equalize this norm for sake of subsequent operators
    upon the coefficients, especially spatial, e.g. convolutions.

    Limitations
    -----------
    At the root of it all is unpadding: aliasing won't play nice.

      - It's crucial to distinguish between "Energy of Signal" (ES) and
        "Energy of Transform" (ET); unpadded CWT conserves former but not latter,
        yet we're interested in latter.
        See: https://dsp.stackexchange.com/a/86182/50076
      - ET loss is indeed info loss. Worst case, consider the unit impulse:
        if placed at center, we keep the full transform along its large scale
        convolutions. If indexed as the left-most sample, we lose an entire half.
        The complete CWT is hence time-expansive, consistent with the uncertainty
        principle. Yet we still unpad, as a heuristic for minimizing coefficient
        "air-packing" - indeed half the transform could be all zero otherwise.
      - The two design objectives are at odds: we cannot both conserve energy
        and maximize information density. So, we prioritize latter, and do our
        best with former.

    Pad-dependence: time & general
    ------------------------------
    Zero-padding is always lossy, for any `x` whose CWT extends outside unpadding.
    Predicting the unpadded energy from padded, compute-efficiently, is difficult
    if not impossible - same of any other pad scheme. Hence,

      > The implementation chooses not to concern itself with unpad correction

    Yet, there's still a forced consideration due to subsampling, described
    in "Overview". Here lies a simple resolution:

        - Energy-amplifying pad: apply fractional unpad index correction
        - Energy-contracting pad: apply said correction *if* we over-unpadded
          (unpadded length < input length). If we under-unpad, we're closer
          to the correct result.
        - `pad_mode='zero'`: since the implementation never over-unpads, we
          never do correction.
        - `pad_mode='reflect'`: is energy-amplifying, do correction.

    Pad-dependence: frequential
    ---------------------------
    More complicated. Of chief importance is to note that, something like
    `np.pad(U_1_slc, amount, 'reflect')` upon e.g. `N_fr=15` while
    `N_frs_max=128`, is *not valid* - since the true continutation of the
    missing input is zero. Shorter `N_fr` are merely the result of discarding
    first-order rows for having too little energy. So, the right-padded portion
    must be zero.

    Left-padding isn't too clear. Arguments:

        - Yes, do `'conj-reflect-zero'`: this can increase FDTS-discriminability
          while also aiding energy conservation.
        - No, do `'zero'`: if we're bandlimited (at least approximately),
          which we must assume for anything to be valid in the first place,
          then the "true" continuation past Nyquist is zero. If we increased
          the sampling rate of `x`, the rows we'd generate would be zero.

    Both make strong points. In fact "Yes" is suggestive of discarding the
    opening paragraph of this section. The resolution might be to simply
    unpad less and make frequential scattering `n1`-expansive.

    Our choice is to not do frequential subsampling-unpad correction, since
    either pad scheme is energy-contracting w.r.t. right-unpadding. That is,
    except an edge case: global averaging.

    Global averaging
    ----------------
    As it's a flat line as a convolution kernel, it collapses all spatial
    information, and is permutation-invariant and doesn't distinguish between
    "padded" and "original". Moreover, subsampling becomes equivalent to
    unpadding, so we're forced to concern with unpad correction.

    On doing fractional unpad correction:

        - Yes: it serves the motivation of energy correction. If padded is `64`
          and unpadded is `37`, and unpadded is "correct", then subsample-only
          correction yields the energy of `64`, not `37`.
        - No: unpadded isn't strictly correct, as discussed. If we only pad by
          one dyadic factor, we expect significant legitimate energy in padded
          region, and this should be reflected in the average.

    We side with "no" for energy-contractive padding, and "yes" for expansive.

    Caveat: `'conj-reflect-zero'` is re-classified as energy-expansive, as the
    "right-unpad-only" point falls apart. Yet, if `2**pad_fr < N_frs_max`,
    `'conj-reflect-zero'` is equivalently `'zero'`. This distinction is coded.
    It also brings forth another limitation.

    Limitations (cont'd)
    --------------------
    For kernels whose supports exceed input's, and require padding to more than
    the next power of 2, the distinction between "global" and local blurs. It's
    an ill-defined and unstudied territory. Hence,

      > The implementation chooses not to concern itself with large-scale
        convolution corrections.

    ("large-scale" as per first sentence).

    Best practice is hence to have `J <= log2(N) - 3` and
    `J_fr <= log2(N_frs_max) - 3` (also for other reasons).

    Lastly, the design choices don't avert `oversampling`-dependence, in that
    `oversampling=inf` avoids any need for fractional index correction, which is
    akin to always doing said correction, yet we often choose not to.

    `out_3D`-dependence
    -------------------
    If we're doing frequential correction, a question is - what is
    `unpad_len_exact`? There's a few perspectives here, but our resolution is
    as follows. Unlike with `average_fr=False` (or just `out_3D=False`), we
    don't compute it as `N_fr / stride`. Instead, `N_frs_max` is treated as
    the ground truth, and we reuse `N_frs_max / stride`.

    Fractional unpad index correction diagram
    -----------------------------------------
    Every case where fractional unpad index correction happens:

    **Time**:

        - `pad_mode != 'zero'` - must have, else pad is energy-contractive

    **Freq**:

        - `pad_mode_fr != 'zero'`
        - `pad_fr > N_fr_scales_max` - must have, else pad mode is `'zero'`
        - `average_fr_global = True` - see "Global averaging"
    """
    # TODO move into compute graph, notable impact on GPU  # TODO wrong
    # TODO only for non-zero pad?
    energy_correction_tm, energy_correction_fr = 1, 1
    if param_tm is not None:
        (do_ec_frac_tm, N, ind_start_tm, ind_end_tm,
         total_conv_stride_tm) = param_tm
        assert do_ec_frac_tm is not None
        if do_ec_frac_tm and ind_start_tm is not None:
            # time energy correction due to integer-rounded unpad indices
            unpad_len_exact = N / 2**total_conv_stride_tm
            unpad_len = ind_end_tm - ind_start_tm
            if not (unpad_len_exact.is_integer() and
                    unpad_len == unpad_len_exact):
                energy_correction_tm *= unpad_len_exact / unpad_len
        # compensate for subsampling
        energy_correction_tm *= 2**total_conv_stride_tm

    if param_fr is not None:
        (do_ec_frac_fr, N_fr, ind_start_fr, ind_end_fr,
         total_conv_stride_over_U1_realized) = param_fr
        assert do_ec_frac_fr is not None
        if do_ec_frac_fr and ind_start_fr is not None:
            # freq energy correction due to integer-rounded unpad indices
            unpad_len_exact = N_fr / 2**total_conv_stride_over_U1_realized
            unpad_len = ind_end_fr - ind_start_fr
            if not (unpad_len_exact.is_integer() and
                    unpad_len == unpad_len_exact):
                energy_correction_fr *= unpad_len_exact / unpad_len
        # compensate for subsampling
        energy_correction_fr *= 2**total_conv_stride_over_U1_realized

        if phi_t_psi_f:
            # since we only did one spin
            energy_correction_fr *= 2

    ec = np.sqrt(energy_correction_tm * energy_correction_fr)
    return ec


def _get_ec_params_fr(scf, n2, out_3D, ind_start_fr, ind_end_fr,
                      total_conv_stride_over_U1_realized):
    pass
    # log2_F_phi_diff, pad_diff, n1_fr_subsample, psi_id = 1

    # if out_3D:
    #     if scf.average_fr_global:
    #         N_fr_extent = scf.N_frs_max
    #     else:
    #         phi_extent = 2 * scf.phi_f_fr['width'][log2_F_phi_diff][pad_diff][
    #             n1_fr_subsample]
    #         psi_extent = 2 * max(scf.psi1_f_fr_up['width'][psi_id])
    #         N_fr_extent = scf.N_frs[n2] + max(phi_extent, psi_extent)
    # else:
    #     N_fr_extent = scf.N_frs[n2]

    # ec = 1
    # unpad_len_exact = N_fr_extent / 2**total_conv_stride_over_U1_realized
    # unpad_len = ind_end_fr - ind_start_fr

    # if out_3D:
    #     pass
    # else:
    #     unpad_excess = bool(unpad_len > unpad_len_exact)
    # N_fr = scf.N_frs[n2]

    # if unpad_len < unpad_len_exact:  # no-cov
    #     raise Exception("This should be impossible.")
    # # TODO revisit width exclude
    # if scf.pad_mode_fr == 'zero':
    #     pad_amplified_energy = False
    #     if out_3D:
    #         if ((ind_end_fr - ind_start_fr) >
    #             N_fr / 2**total_conv_stride_over_U1_realized):
    #             # excess + unamplified -> unpad is closer to the complete result
    #             pass
    #     else:
    #         if unpad_excess:
    #             # excess + unamplified -> unpad it closer to the complete result
    #             pass
    # else:
    #     pad_amplified_energy = bool(N_fr_extent > 2*scf.N_frs_max)
    #     if out_3D:
    #         if ((ind_end_fr - ind_start_fr) >
    #             N_fr / 2**total_conv_stride_over_U1_realized):
    #             if pad_amplified_energy:
    #                 unpad_len_exact = (N_fr_extent /
    #                                    2**total_conv_stride_over_U1_realized)
    #                 unpad_len = ind_end_fr - ind_start_fr
    #                 ec *= unpad_len_exact / unpad_len
    #             pass
    #     else:
    #         # Here we're always within a stride of `N_fr`.
    #         # `'reflect'`-amplified energy means we treat `N_fr` as the ground
    #         # truth
    #         if unpad_excess:
    #             if pad_amplified_energy:
    #                 # excess + amplified -> attenuate
    #                 ec *= unpad_len_exact / unpad_len
    #             else:
    #                 # excess + unamplified -> unpad is closer to the complete
    #                 # result
    #                 pass

    # if scf.pad_mode_fr == 'custom':  # TODO nope, don't
    #     warnings.warn("Custom `pad_mode_fr` reuses `'conj-reflect-zero'`'s "
    #                   "energy correction, which may not be ideal.")

    # param_fr = (N_fr_extent, ind_start_fr,
    #             ind_end_fr, total_conv_stride_over_U1_realized)

# metas ######################################################################
def compute_meta_scattering(psi1_f, psi2_f, phi_f, log2_T, paths_include_n2n1,
                            max_order=2):
    """Get metadata of the Wavelet Time Scattering transform.

    Specifies the content of each scattering coefficient - which order,
    frequencies, filters were used, and so on. See below for more info.

    See `help(wavespin.Scattering1D())` for description of parameters.

    Returns
    -------
    meta : dictionary
        A dictionary with the following keys:

        - `'order`' : tensor
            A tensor of length `C`, the total number of scattering
            coefficients, specifying the scattering order.
        - `'xi'` : tensor
            A tensor of size `(C, max_order)`, specifying the center
            frequency of the filter used at each order (padded with NaNs).
        - `'sigma'` : tensor
            A tensor of size `(C, max_order)`, specifying the frequency
            bandwidth of the filter used at each order (padded with NaNs).
        - `'j'` : tensor
            A tensor of size `(C, max_order)`, specifying the dyadic scale
            of the filter used at each order (padded with NaNs).
        - `'is_cqt'` : tensor
            A tensor of size `(C, max_order)`, specifying whether the filter
            was constructed per Constant Q Transform (padded with NaNs).
        - `'n'` : tensor
            A tensor of size `(C, max_order)`, specifying the indices of
            the filters used at each order (padded with NaNs).
        - `'key'` : list
            The tuples indexing the corresponding scattering coefficient
            in the non-vectorized output.

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


def compute_meta_jtfs(scf, psi1_f, psi2_f, phi_f, log2_T, sigma0,
                      average, average_global, average_global_phi,
                      oversampling, out_type, out_exclude, paths_exclude,
                      api_pair_order):
    """Get metadata of the Joint Time-Frequency Scattering transform.

    Specifies the content of each scattering coefficient - which order,
    frequencies, filters were used, and so on. See below for more info.

    Parameters
    ----------
    scf : `scattering1d.frontend.base_frontend._FrequencyScatteringBase1D`
        Frequential scattering object, storing pertinent attributes and filters.

    psi1_f, psi2_f, phi_f : list, list, list
        See `help(wavespin.scattering1d.TimeFrequencyScattering1D)`.
        Meta for time scattering is extracted directly from filters.

    log2_T, sigma0: int, float
        See `help(wavespin.scattering1d.TimeFrequencyScattering1D)`.

    average : bool
        Affects `S0`'s meta, and temporal stride meta.

    average_global : bool
        Affects `S0`'s meta, and temporal stride meta.

    average_global_phi : bool
        Affects joint temporal stride meta.

    oversampling : int
        See `help(wavespin.scattering1d.TimeFrequencyScattering1D)`.
        Affects temporal stride meta.

    out_type : str
         - `'dict:list'` or `'dict:array'`: meta is packed
           into respective pairs (e.g. `meta['n']['psi_t * phi_f'][1]`)
         - `'list'` or `'array'`: meta is flattened (e.g. `meta['n'][15]`).

    out_exclude : list/tuple[str]
        Names of coefficient pairs to exclude from meta.

    paths_exclude : dict / None
        See `help(wavespin.scattering1d.TimeFrequencyScattering1D)`.
        Paths to exclude from meta.

    api_pair_order : set[str]
        To follow the API ordering, alongside coefficients.

    Returns
    -------
    meta : dictionary
        Each value is a tensor, `C` is the total number of scattering coeffs,
        and each tensor is padded with NaNs where appropraite. Key `'key'` is
        an exception, which is a list.

        - `'order`' : length `C`
            Scattering order.

        - `'xi'` : shape `(C, 3)`
            Center frequency of the filter used.

        - `'sigma'` : shape `(C, 3)`
            Bandwidth of the filter used as a continuous-time parameter.

        - `'j'` : shape `(C, 3)`
            Maximum permitted subsampling factor of the filter used.

        - `'is_cqt'` : shape `(C, 3)`
            `True` if the filter was constructed per Constant Q Transform.

        - `'n'` : shape `(C, 3)`
            Indices of the filters used.
            Lowpass filters in `phi_*` pairs are denoted via `-1`.

        - `'spin'` : length `C`
            Spin of each frequential scattering filter used.
            +1=up, -1=down, 0=none.

        - `'slope'` : length `C`
            Slope of each frequential scattering filter used, in wavelets/sample.
            To convert to

               - octaves/sample:  `slope /= Q1`, `Q1 = Q[0]`.
               - wavelets/second: `slope *= fs`, `fs` is sampling rate in `Hz`.
               - octaves/second:  `slope *= (fs / Q1)`.

            NOTE: not to be interpreted as pure slope or rotation, instead as
            a product of different relative scalings along time and log-frequency.

        - `'stride'` : shape `(C, 2)`
            A tensor of size `(C, 2)`, specifying the total temporal and
            frequential convolutional stride (i.e. subsampling) of resulting
            coefficient (including lowpass filtering).

        - `'key'` : list
            `slice`'s for converting between coefficient and meta indices.
            See "Coefficients <-> meta" below.

        In case of `out_3D=True`, for joint pairs, will reshape each field into
        `(n_coeffs, C, meta_len)`, where `n_coeffs` is the number of joint slices
        in the pair, and `meta_len` is the existing `shape[-1]` (1, 2, or 3).

        Tensors sized `(..., 3)` index dim 1 as `(n2, n1_fr, n1)`, so e.g.

            - `out_3D=False`: `meta['n'][5, 0]` fetches `n2` for coefficient
              indexed with 5, so the second-order filter used to produce that
              coefficient is `psi2_f[n2]`.
            - `out_3D=True`: `meta['n'][0, 5, 0]` accomplishes same as above.
              `meta['n'][5, 0, 1]` here fetches the `n1_fr` of the first
              coefficient of the sixth joint slice.

    Coefficients <-> meta
    ---------------------
    Meta is built such that indexing meta equates indexing coefficients for
    'array' `out_type`, and same for 'list' minus the ability to index joint
    coeffs with `n1` (also 'dict:' for each). Hence conversion is straightforward,
    but takes some data reformatting to achieve; this is facilitated by the
    `'key'` meta.

    See `wavespin.toolkit.coeff2meta_jtfs()` and
    `wavespin.toolkit.meta2coeff_jtfs()`.

    Computation and Structure
    -------------------------
    Computation replicates logic in `timefrequency_scattering1d()`. Meta values
    depend on:

        - out_3D (True only possible with `average and average_fr`)
        - aligned
        - sampling_psi_fr
        - sampling_phi_fr
        - average
        - average_global
        - average_global_phi
        - average_fr
        - average_fr_global
        - average_fr_global_phi
        - oversampling
        - oversampling_fr
        - max_pad_factor_fr (mainly via `unrestricted_pad_fr`)
        - max_noncqt_fr
        - out_exclude
        - paths_exclude
        - paths_include_build (not user-set)

    and some of their interactions. Listed are only "unobvious" parameters;
    anything that controls the filterbanks will change meta (`J`, `Q`, etc).
    """  # TODO
    def _get_compute_params(n2, n1_fr):
        """Reproduce the exact logic in `timefrequency_scattering1d.py`."""
        # basics
        scale_diff = scf.scale_diffs[n2]
        J_pad_fr = scf.J_pad_frs[scale_diff]
        N_fr_padded = 2**J_pad_fr

        # n1_fr_subsample, lowpass_subsample_fr ##############################
        global_averaged_fr = (scf.average_fr_global if n1_fr != -1 else
                              scf.average_fr_global_phi)
        if n2 == -1 and n1_fr == -1:
            lowpass_subsample_fr = 0
            if scf.average_fr_global_phi:
                n1_fr_subsample = scf.log2_F
                log2_F_phi = scf.log2_F
                log2_F_phi_diff = 0
            else:
                log2_F_phi = scf.log2_F_phis['phi'][scale_diff]
                log2_F_phi_diff = scf.log2_F_phi_diffs['phi'][scale_diff]
                n1_fr_subsample = max(scf.n1_fr_subsamples['phi'][scale_diff] -
                                      scf.oversampling_fr, 0)

        elif n1_fr == -1:
            lowpass_subsample_fr = 0
            if scf.average_fr_global_phi:
                total_conv_stride_over_U1_phi = min(J_pad_fr, scf.log2_F)
                n1_fr_subsample = total_conv_stride_over_U1_phi
                log2_F_phi = scf.log2_F
                log2_F_phi_diff = 0
            else:
                n1_fr_subsample = max(scf.n1_fr_subsamples['phi'][scale_diff] -
                                      scf.oversampling_fr, 0)
                log2_F_phi = scf.log2_F_phis['phi'][scale_diff]
                log2_F_phi_diff = scf.log2_F_phi_diffs['phi'][scale_diff]

        else:
            total_conv_stride_over_U1 = (
                scf.total_conv_stride_over_U1s[scale_diff][n1_fr])
            n1_fr_subsample = max(scf.n1_fr_subsamples['spinned'
                                                       ][scale_diff][n1_fr] -
                                  scf.oversampling_fr, 0)
            log2_F_phi = scf.log2_F_phis['spinned'][scale_diff][n1_fr]
            log2_F_phi_diff = scf.log2_F_phi_diffs['spinned'][scale_diff][n1_fr]
            if global_averaged_fr:
                lowpass_subsample_fr = (total_conv_stride_over_U1 -
                                        n1_fr_subsample)
            elif scf.average_fr:
                lowpass_subsample_fr = max(total_conv_stride_over_U1 -
                                           n1_fr_subsample -
                                           scf.oversampling_fr, 0)
            else:
                lowpass_subsample_fr = 0

        # total stride, unpadding ############################################
        total_conv_stride_over_U1_realized = (n1_fr_subsample +
                                              lowpass_subsample_fr)

        if scf.out_3D:
            _stride_ref = scf.total_conv_stride_over_U1s[0][0]
            stride_ref = max(_stride_ref - scf.oversampling_fr, 0)
            ind_start_fr = scf.ind_start_fr_max[stride_ref]
            ind_end_fr   = scf.ind_end_fr_max[  stride_ref]
        else:
            _stride = total_conv_stride_over_U1_realized
            ind_start_fr = scf.ind_start_fr[n2][_stride]
            ind_end_fr   = scf.ind_end_fr[  n2][_stride]

        return (N_fr_padded, total_conv_stride_over_U1_realized,
                n1_fr_subsample, scale_diff, log2_F_phi_diff, log2_F_phi,
                ind_start_fr, ind_end_fr, global_averaged_fr)

    def _get_fr_params(n1_fr, scale_diff, log2_F_phi_diff, log2_F_phi):
        if n1_fr != -1:
            # spinned
            psi_id = scf.psi_ids[scale_diff]
            p = [scf.psi1_f_fr_up[field][psi_id][n1_fr]
                 for field in ('xi', 'sigma', 'j', 'is_cqt')]
        else:
            # phi_f
            if not scf.average_fr_global:
                F_phi = scf.F / 2**log2_F_phi_diff
                p = (0., sigma0 / F_phi, log2_F_phi, nan)
            else:
                p = (0., sigma0 / 2**log2_F_phi, log2_F_phi, nan)

        xi1_fr, sigma1_fr, j1_fr, is_cqt1_fr = p
        return xi1_fr, sigma1_fr, j1_fr, is_cqt1_fr

    def _exclude_excess_scale(n2, n1_fr):
        scale_diff = scf.scale_diffs[n2]
        psi_id = scf.psi_ids[scale_diff]
        j1_frs = scf.psi1_f_fr_up['j'][psi_id]
        return bool(n1_fr > len(j1_frs) - 1)

    def _get_slope(n2, n1_fr):
        if n2 == -1 and n1_fr == -1:
            slope = nan
        elif n2 == -1:
            slope = 0
        elif n1_fr == -1:
            slope = math.inf
        else:
            # fetch peaks from actual wavelets used
            psi_id = scf.psi_ids[scf.scale_diffs[n2]]
            peak1_fr = scf.psi1_f_fr_up['peak_idx'][psi_id][n1_fr]

            # compute w.r.t. original, i.e. unsubsampled, untrimmed quantities
            # (signal, scalogram, wavelet), i.e. max oversampling
            # for time this means sampling rate
            N2 = len(psi2_f[n2][0])
            # for freq this means length, meaning scale to the wavelet
            N1_fr = len(scf.psi1_f_fr_up[psi_id][0])

            # note peak index is independent of subsampling factor
            peak2 = psi2_f[n2]['peak_idx'][0]

            # compute & normalize such that slope is measured in [samples]
            slope = (peak2 / N2) / (peak1_fr / N1_fr)

        return slope

    def _skip_path(n2, n1_fr):
        # n2 >= 0 since build won't have `-1` as a key and won't skip phi_t path
        build_skip_path = bool(n2 >= 0 and n2 not in scf.paths_include_build)
        if build_skip_path:
            # can't do other checks and don't need to
            return True

        excess_scale = bool(scf.sampling_psi_fr == 'exclude' and
                            _exclude_excess_scale(n2, n1_fr))
        user_skip_path = bool(n2 in paths_exclude.get('n2', {}) or
                              n1_fr in paths_exclude.get('n1_fr', {}))
        return excess_scale or user_skip_path

    def _fill_n1_info(pair, n2, n1_fr, spin):
        if _skip_path(n2, n1_fr):
            return

        # track S1 from padding to `_joint_lowpass()`
        (N_fr_padded, total_conv_stride_over_U1_realized, n1_fr_subsample,
         scale_diff, log2_F_phi_diff, log2_F_phi, ind_start_fr, ind_end_fr,
         global_averaged_fr) = _get_compute_params(n2, n1_fr)

        # fetch xi, sigma for n2, n1_fr
        if n2 != -1:
            xi2, sigma2, j2, is_cqt2 = (xi2s[n2], sigma2s[n2], j2s[n2],
                                        is_cqt2s[n2])
        else:
            xi2, sigma2, j2, is_cqt2 = 0., sigma_low, log2_T, nan
        xi1_fr, sigma1_fr, j1_fr, is_cqt1_fr = _get_fr_params(
            n1_fr, scale_diff, log2_F_phi_diff, log2_F_phi)

        # get temporal stride info
        global_averaged = (average_global if n2 != -1 else
                           average_global_phi)
        if global_averaged:
            total_conv_stride_tm = log2_T
        else:
            k1_plus_k2 = max(min(j2, log2_T) - oversampling, 0)
            if average:
                k2_tm_J = max(log2_T - k1_plus_k2 - oversampling, 0)
                total_conv_stride_tm = k1_plus_k2 + k2_tm_J
            else:
                total_conv_stride_tm = k1_plus_k2
        stride = (total_conv_stride_over_U1_realized, total_conv_stride_tm)

        # convert to filter format but preserve original for remaining meta steps
        n1_fr_n = n1_fr if (n1_fr != -1) else inf
        n2_n    = n2    if (n2    != -1) else inf

        # find slope
        slope = _get_slope(n2, n1_fr)

        # global average pooling, all S1 collapsed into single point
        if global_averaged_fr:
            meta['order' ][pair].append(2)
            meta['xi'    ][pair].append((xi2,     xi1_fr,     nan))
            meta['sigma' ][pair].append((sigma2,  sigma1_fr,  nan))
            meta['j'     ][pair].append((j2,      j1_fr,      nan))
            meta['is_cqt'][pair].append((is_cqt2, is_cqt1_fr, nan))
            meta['n'     ][pair].append((n2_n,    n1_fr_n,    nan))
            meta['spin'  ][pair].append(spin)
            meta['stride'][pair].append(stride)
            meta['slope' ][pair].append(slope)
            if pair not in out_exclude:
                meta['key'][pair].append(slice(rkey[:], rkey[:] + 1))
                rkey[:] += 1
            return

        fr_max = scf.N_frs[n2] if (n2 != -1) else len(xi1s)
        n_n1s = 0

        # account for repadding
        if ind_end_fr - ind_start_fr > N_fr_padded:
            assert scf.out_3D and scf.oversampling_fr > 0
            N_fr_max = 2**math.ceil(math.log2(ind_end_fr - ind_start_fr))
        else:
            N_fr_max = N_fr_padded

        # simulate subsampling
        n1_step = 2 ** total_conv_stride_over_U1_realized
        for n1 in range(0, N_fr_max, n1_step):
            # simulate unpadding
            if n1 / n1_step < ind_start_fr:
                continue
            elif n1 / n1_step >= ind_end_fr:
                break

            if n1 >= fr_max:
                # these are padded rows, no associated filters
                xi1, sigma1, j1, is_cqt1 = nan, nan, nan, nan
            else:
                xi1, sigma1, j1, is_cqt1 = (xi1s[n1], sigma1s[n1], j1s[n1],
                                            is_cqt1s[n1])
            meta['order' ][pair].append(2)
            meta['xi'    ][pair].append((xi2,     xi1_fr,     xi1))
            meta['sigma' ][pair].append((sigma2,  sigma1_fr,  sigma1))
            meta['j'     ][pair].append((j2,      j1_fr,      j1))
            meta['is_cqt'][pair].append((is_cqt2, is_cqt1_fr, is_cqt1))
            meta['n'     ][pair].append((n2_n,    n1_fr_n,    n1))
            meta['spin'  ][pair].append(spin)
            meta['stride'][pair].append(stride)
            meta['slope' ][pair].append(slope)
            n_n1s += 1

        if pair not in out_exclude:
            start, end = rkey[:], rkey[:] + n_n1s
            if 'array' in out_type:
                meta['key'][pair].extend([slice(i, i + 1)
                                          for i in range(start, end)])
            else:
                meta['key'][pair].append(slice(start, end))
            rkey[:] += n_n1s

    # handle `out_exclude`
    if out_exclude is None:
        out_exclude = ()

    # extract meta
    log2_F = scf.log2_F
    sigma_low = phi_f['sigma']

    j1_frs = scf.psi1_f_fr_up['j']
    fields = ('xi', 'sigma', 'j', 'is_cqt')
    meta1, meta2 = [{field: [p[field] for p in psi_f] for field in fields}
                    for psi_f in (psi1_f, psi2_f)]
    xi1s, sigma1s, j1s, is_cqt1s = meta1.values()
    xi2s, sigma2s, j2s, is_cqt2s = meta2.values()

    # fetch phi meta; must access `phi_f_fr` as `j1s_fr` requires sampling phi
    meta_phi = {}
    for field in ('xi', 'sigma', 'j'):
        meta_phi[field] = {}
        for k in scf.phi_f_fr[field]:
            meta_phi[field][k] = scf.phi_f_fr[field][k]
    xi1s_fr_phi, sigma1_fr_phi, j1s_fr_phi = list(meta_phi.values())

    meta = {}
    inf = -1  # placeholder for infinity
    nan = math.nan
    # this order must match, for `'dict' not in out_type`, what's used in
    # `timefrequency_scattering1d.py`
    coef_names = (
        'S0',                # (time)  zeroth order
        'S1',                # (time)  first order
        'phi_t * phi_f',     # (joint) joint lowpass
        'phi_t * psi_f',     # (joint) time lowpass
        'psi_t * phi_f',     # (joint) freq lowpass
        'psi_t * psi_f_up',  # (joint) spin up
        'psi_t * psi_f_dn',  # (joint) spin down
    )
    for field in ('order', 'xi', 'sigma', 'j', 'is_cqt', 'n', 'spin', 'stride',
                  'slope', 'key'):
        meta[field] = {name: [] for name in coef_names}

    # for 'key' meta to increment throughout pairs
    class RunningKey():
        def __init__(self):
            self.count = 0

        def __getitem__(self, _):
            return self.count

        def __setitem__(self, _, value):
            self.count = value

        def maybe_reset(self):
            if 'dict' in out_type:
                self.count = 0

    rkey = RunningKey()

    # Zeroth-order ###########################################################
    if average_global:
        k0 = log2_T
    elif average:
        k0 = max(log2_T - oversampling, 0)
    meta['order' ]['S0'].append(0)
    meta['xi'    ]['S0'].append((nan, nan, 0.        if average else nan))
    meta['sigma' ]['S0'].append((nan, nan, sigma_low if average else nan))
    meta['j'     ]['S0'].append((nan, nan, log2_T    if average else nan))
    meta['is_cqt']['S0'].append((nan, nan, nan))
    meta['n'     ]['S0'].append((nan, nan, inf       if average else nan))
    meta['spin'  ]['S0'].append(nan)
    meta['stride']['S0'].append((nan, k0 if average else nan))
    meta['slope' ]['S0'].append(nan)
    if 'S0' not in out_exclude:
        meta['key']['S0'].append(slice(rkey[:], rkey[:] + 1))
        rkey[:] += 1

    # First-order ############################################################
    def stride_S1(j1):
        k1 = max(min(j1, log2_T) - oversampling, 0)
        k1_log2_T = max(log2_T - k1 - oversampling, 0)

        if average_global:
            total_conv_stride_tm = log2_T
        elif average:
            total_conv_stride_tm = k1 + k1_log2_T
        else:
            total_conv_stride_tm = k1
        return total_conv_stride_tm

    rkey[:] = 0 if out_type.startswith('dict') else rkey[:]
    for (n1, (xi1, sigma1, j1, is_cqt1)
         ) in enumerate(zip(xi1s, sigma1s, j1s, is_cqt1s)):
        meta['order' ]['S1'].append(1)
        meta['xi'    ]['S1'].append((nan, nan, xi1))
        meta['sigma' ]['S1'].append((nan, nan, sigma1))
        meta['j'     ]['S1'].append((nan, nan, j1))
        meta['is_cqt']['S1'].append((nan, nan, is_cqt1))
        meta['n'     ]['S1'].append((nan, nan, n1))
        meta['spin'  ]['S1'].append(nan)
        meta['stride']['S1'].append((nan, stride_S1(j1)))
        meta['slope']['S1'].append(nan)
        if 'S1' not in out_exclude:
            meta['key']['S1'].append(slice(rkey[:], rkey[:] + 1))
            rkey[:] += 1

    S1_len = len(meta['n']['S1'])
    assert S1_len >= scf.N_frs_max, (S1_len, scf.N_frs_max)

    # Joint scattering #######################################################
    # `phi_t * phi_f` coeffs
    rkey.maybe_reset()
    _fill_n1_info('phi_t * phi_f', n2=-1, n1_fr=-1, spin=0)

    # `phi_t * psi_f` coeffs
    rkey.maybe_reset()
    for n1_fr in range(len(j1_frs[0])):
        _fill_n1_info('phi_t * psi_f', n2=-1, n1_fr=n1_fr, spin=0)

    # `psi_t * phi_f` coeffs
    rkey.maybe_reset()
    for n2 in range(len(j2s)):
        if n2 in scf.paths_include_build:
            _fill_n1_info('psi_t * phi_f', n2, n1_fr=-1, spin=0)

    # `psi_t * psi_f` coeffs
    for spin in (1, -1):
        pair = ('psi_t * psi_f_up' if spin == 1 else
                'psi_t * psi_f_dn')
        rkey.maybe_reset()
        for n2 in range(len(j2s)):
            if n2 in scf.paths_include_build:
                psi_id = scf.psi_ids[scf.scale_diffs[n2]]
                for n1_fr, j1_fr in enumerate(j1_frs[psi_id]):
                    _fill_n1_info(pair, n2, n1_fr, spin=spin)

    # convert to array
    array_fields = ['order', 'xi', 'sigma', 'j', 'is_cqt', 'n', 'spin', 'slope',
                    'stride']
    for field in array_fields:
        for pair, v in meta[field].items():
            meta[field][pair] = np.array(v)

    # handle `out_3D` ########################################################
    if scf.out_3D:
        # reorder for 3D
        for field in array_fields:
            # meta_len
            if field in ('spin', 'slope', 'order'):
                meta_len = 1
            elif field == 'stride':
                meta_len = 2
            else:
                meta_len = 3

            for pair in meta[field]:
                # number of n2s
                if pair.startswith('phi_t'):
                    n_n2s = 1
                else:
                    n_n2s = sum((n2 in scf.paths_include_build and
                                 n2 not in paths_exclude['n2'])
                                for n2 in range(len(j2s)))

                # number of n1_frs; n_slices
                n_slices = None
                if pair in ('S0', 'S1'):
                    # simply expand dim for consistency, no 3D structure
                    meta[field][pair] = meta[field][pair].reshape(-1, 1, meta_len)
                    continue

                elif 'psi_f' in pair:
                    if pair.startswith('phi_t'):
                        n_slices = sum(not _skip_path(n2=-1, n1_fr=n1_fr)
                                       for n1_fr in range(len(j1_frs[0])))
                    else:
                        n_slices = sum(not _skip_path(n2=n2, n1_fr=n1_fr)
                                       for n2 in range(len(j2s))
                                       for n1_fr in range(len(j1_frs[0])))

                elif 'phi_f' in pair:
                    n_n1_frs = 1

                # n_slices
                if n_slices is None:
                    n_slices = n_n2s * n_n1_frs

                    # reshape meta
                shape = ((n_slices, -1, meta_len) if meta_len != 1 else
                         (n_slices, -1))
                meta[field][pair] = meta[field][pair].reshape(shape)

        # sanity check: each dim0 index must have same `n2` & `n1_fr`
        for pair in meta['n']:
            if pair in ('S0', 'S1'):
                continue
            for i in range(len(meta['n'][pair])):
                n2s_n1_frs = meta['n'][pair][i, :, :2]
                assert np.max(np.diff(n2s_n1_frs, axis=0)) == 0, n2s_n1_frs

    # handle `out_exclude ####################################################
    # first check that we followed the API pair order
    for field in meta:
        assert tuple(meta[field]) == api_pair_order, (
            tuple(meta['field']), api_pair_order)

    if out_exclude:
        # drop excluded pairs
        for pair in out_exclude:
            for field in meta:
                del meta[field][pair]

    # sanity checks ##########################################################
    # ensure time / freq stride doesn't exceed log2_T / log2_F in averaged cases,
    # and realized J / J_fr in unaveraged
    max_j1, max_j2 = max(j1s), max(j2s)
    max_joint_tm_j = max_j2  # joint coeff bandpass stride controlled by j2 alone
    max_j_fr = max(j1_frs[0])
    smax_t_nophi = log2_T if average else max_joint_tm_j
    if scf.average_fr:
        if not scf.out_3D and not scf.aligned:
            # see "Compute logic: stride, padding" in `core`
            smax_f_nophi = max(scf.log2_F, max_j_fr)
        else:
            smax_f_nophi = scf.log2_F
    else:
        smax_f_nophi = max_j_fr
    for pair in meta['stride']:
        if pair == 'S0' and not average:
            continue
        if pair != 'S1':
            stride_max_t = (smax_t_nophi if ('phi_t' not in pair) else
                            log2_T)
            stride_max_f = (smax_f_nophi if ('phi_f' not in pair) else
                            log2_F)
        else:
            stride_max_t = log2_T if average else max_j1

        for i, s in enumerate(meta['stride'][pair][..., 1].ravel()):
            assert s <= stride_max_t, ("meta['stride'][{}][{}] > stride_max_t "
                                       "({} > {})").format(pair, i, s,
                                                           stride_max_t)
        if pair in ('S0', 'S1'):
            continue
        for i, s in enumerate(meta['stride'][pair][..., 0].ravel()):
            assert s <= stride_max_f, ("meta['stride'][{}][{}] > stride_max_f "
                                       "({} > {})").format(pair, i, s,
                                                           stride_max_f)

    # handle `out_type #######################################################
    if not out_type.startswith('dict'):
        # join pairs
        if not scf.out_3D:
            meta_flat = {}
            for field in meta:
                vals = list(meta[field].values())
                if field != 'key':
                    vals = np.concatenate(vals, axis=0)
                else:
                    vals = [v for vpair in vals for v in vpair]
                meta_flat[field] = vals
        else:
            meta_flat0, meta_flat1 = {}, {}
            for field in meta:
                vals0 = [v for pair in meta[field] for v in meta[field][pair]
                         if pair in ('S0', 'S1')]
                vals1 = [v for pair in meta[field] for v in meta[field][pair]
                         if pair not in ('S0', 'S1')]
                if field != 'key':
                    vals0, vals1 = [np.array(v) for v in (vals0, vals1)]
                    vals0 = vals0.squeeze(1)
                meta_flat0[field], meta_flat1[field] = vals0, vals1

            meta_flat = (meta_flat0, meta_flat1)
        meta = meta_flat

    return meta
