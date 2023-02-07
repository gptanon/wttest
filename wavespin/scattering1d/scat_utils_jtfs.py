# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import numpy as np
import math

from .filter_bank import N_and_pad_2_J_pad
from ..utils.gen_utils import npy


# Graph builders #############################################################
def build_compute_graph_fr(self):
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
        DT[n2].update(
            j2=j2,
            k1_plus_k2=k1_plus_k2,
            pad_fr=pad_fr,
            **maybe_unpad_time_info,
        )

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

    DT[n2].update(
        j2=j2,
        k1_plus_k2=k1_plus_k2,
        trim_tm=trim_tm,
        pad_fr=pad_fr,
    )
    _compute_graph_frequency_scattering(n2, pad_fr, DT, DF, DL, self)

    # Determine `n1_fr`-dependence of `_joint_lowpass_part2()` ##############
    for n2 in DF:
        part2_grouped_by_n1_fr_subsample = True
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
    DF[n2].update(
        psi_id=psi_id,
        pad_diff=pad_diff,
        n1_fr_subsamples=n1_fr_subsamples,
        log2_F_phi_diffs=log2_F_phi_diffs,
        unpad_early_fr=unpad_early_fr,
    )

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
            'n_n1s_pre_part2'] = n_n1s


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
        info.update(
            energy_correction=energy_correction,
            param_tm=param_tm,
            param_fr=param_fr,
        )
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
    info.update(trim_tm=trim_tm,
                do_unpad=do_unpad,
                do_unpad_dyadic=do_unpad_dyadic)
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
        Dearly['S1'].update(
            ind_start_tm_avg=ind_start_tm_avg,
            ind_end_tm_avg=ind_end_tm_avg,
            total_conv_stride_tm_avg=total_conv_stride_tm_avg,
        )
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
            Dearly['phi_t * phi_f'].update(
                log2_F_phi_diff=log2_F_phi_diff,
                pad_diff=pad_diff,
            )

    # return
    return Dearly


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

# Runtime helpers ############################################################
def make_psi1_f_fr_stacked_dict(scf, paths_exclude):
    oversampling_fr = scf.oversampling_fr
    # first clear the existing attribute, for memory
    scf.psi1_f_fr_stacked_dict = {}

    psi1_f_fr_packed_dict = {}
    for scale_diff in scf.scale_diffs_unique:
        if scale_diff in psi1_f_fr_packed_dict:
            continue
        psi1_f_fr_packed_dict[scale_diff] = {}
        n1_fr_subsamples = scf.n1_fr_subsamples['spinned'][scale_diff]
        psi_id = scf.psi_ids[scale_diff]

        if scf.vectorized_early_fr:
            # pack all together
            if not paths_exclude['n1_fr']:
                # this is just here for readability
                psi1_f_fr_packed_dict[scale_diff] = (scf.psi1_f_fr_up[psi_id],
                                                      scf.psi1_f_fr_dn[psi_id])
            else:
                ups, dns = [
                    [pf for n1_fr, pf in enumerate(psi1_f_frs[psi_id])
                     if n1_fr not in paths_exclude['n1_fr']]
                    for psi1_f_frs in (scf.psi1_f_fr_up, scf.psi1_f_fr_dn)
                ]
                psi1_f_fr_packed_dict[scale_diff] = (ups, dns)
        else:
            # group by `n1_fr_subsample`
            for n1_fr in range(len(scf.psi1_f_fr_up[psi_id])):
                if n1_fr in paths_exclude['n1_fr']:
                    continue
                n1_fr_subsample = max(n1_fr_subsamples[n1_fr] -
                                      oversampling_fr, 0)
                # append for stacking
                if n1_fr_subsample not in psi1_f_fr_packed_dict[scale_diff]:
                    psi1_f_fr_packed_dict[scale_diff][n1_fr_subsample] = ([], [])
                psi1_f_fr_packed_dict[scale_diff][n1_fr_subsample][0].append(
                    scf.psi1_f_fr_up[psi_id][n1_fr])
                psi1_f_fr_packed_dict[scale_diff][n1_fr_subsample][1].append(
                    scf.psi1_f_fr_dn[psi_id][n1_fr])

    # track duplicates -------------------------------------------------------
    def _are_equal(p0s, p1s, equals):
        # assert expected structures
        assert isinstance(p0s, (dict, list, tuple)) or hasattr(p0s, 'ndim')
        if len(p0s) != len(p1s):
            equals.append(False)
            return

        if isinstance(p0s, dict):
            p0s, p1s = p0s.values(), p1s.values()

        for elem0, elem1 in zip(p0s, p1s):
            if len(elem0) != len(elem1):
                equals.append(False)
                return
            if hasattr(elem0, 'ndim'):
                equals.append(id(elem0) == id(elem1))
            else:
                _are_equal(elem0, elem1, equals)

    def are_equal(stacked_scale, stacked_scale_next):
        equals = []
        p0s, p1s = stacked_scale, stacked_scale_next
        _are_equal(p0s, p1s, equals)
        return all(equals)

    scale_diffs = sorted(list(psi1_f_fr_packed_dict))
    scale_diffs_duplicate = {}
    for i, scale_diff in enumerate(scale_diffs):
        if i == len(scale_diffs) - 1:
            break
        scale_diff_next = scale_diffs[i + 1]
        stacked_scale = psi1_f_fr_packed_dict[scale_diff]
        stacked_scale_next = psi1_f_fr_packed_dict[scale_diff_next]

        if are_equal(stacked_scale, stacked_scale_next):
            assert (scf.n1_fr_subsamples['spinned'][scale_diff] ==
                    scf.n1_fr_subsamples['spinned'][scale_diff_next])
            scale_diffs_duplicate[scale_diff_next] = scale_diff
    # ------------------------------------------------------------------------

    # do stacking
    psi1_f_fr_stacked_dict = {}
    stack_ids = {}
    stack_id = 0
    stack_id_prev = None
    for scale_diff in psi1_f_fr_packed_dict:
        # if duplicate, re-referene via `stack_id`
        if scale_diff in scale_diffs_duplicate:
            # cannot be the very first scale
            assert scale_diff != list(psi1_f_fr_stacked_dict)[0]
            stack_ids[scale_diff] = stack_id_prev
            continue

        # broadcast along batch dim
        if scf.vectorized_early_fr:
            psi1_f_fr_stacked_dict[stack_id] = np.stack(
                npy(psi1_f_fr_packed_dict[scale_diff])
                )[None]
        else:
            psi1_f_fr_stacked_dict[stack_id] = {}
            for n1_fr_subsample in psi1_f_fr_packed_dict[scale_diff]:
                psi1_f_fr_stacked_dict[stack_id][n1_fr_subsample] = np.stack(
                    npy(psi1_f_fr_packed_dict[scale_diff][n1_fr_subsample])
                    )[None]

        # update `stack_id`
        stack_ids[scale_diff] = stack_id
        stack_id_prev = stack_id
        stack_id += 1

    return psi1_f_fr_stacked_dict, stack_ids


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
            stack_id = scf.stack_ids[scale_diff]
            if scf.vectorized_early_fr:
                assert (stacked[stack_id].shape[2] ==
                        sum(map(len, Y_1_fr_dict[scale_diff].values())))
            else:
                for n1_fr_subsample in Y_1_fr_dict[scale_diff]:
                    assert (stacked[stack_id][n1_fr_subsample].shape[2] ==
                            len(Y_1_fr_dict[scale_diff][n1_fr_subsample]))

    # Determine the spins to be computed -------------------------------------
    spin_data = {}
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

        for scale_diff in scf.scale_diffs_unique:
            if not scf.vectorized_fr:
                psi_id = scf.psi_ids[scale_diff]
                psi1_f_frs = []
                if 1 in spins or 0 in spins:
                    psi1_f_frs.append(scf.psi1_f_fr_dn[psi_id])
                if -1 in spins:
                    psi1_f_frs.append(scf.psi1_f_fr_up[psi_id])
            else:
                stack_id = scf.stack_ids[scale_diff]
                stacked_scale = scf.psi1_f_fr_stacked_dict[stack_id]
                if len(spins) == 2:
                    if scf.vectorized_early_fr:
                        psi1_f_fr_stacked = stacked_scale
                    else:
                        psi1_f_fr_stacked_subdict = stacked_scale
                else:
                    s_idx = (1 if (1 in spins or 0 in spins) else
                             0)
                    if scf.vectorized_early_fr:
                        psi1_f_fr_stacked = stacked_scale[:, s_idx:s_idx + 1]
                    else:
                        psi1_f_fr_stacked_subdict = {}
                        for n1_fr_subsample in stacked_scale:
                            ss_sub = stacked_scale[n1_fr_subsample]
                            psi1_f_fr_stacked_subdict[
                                n1_fr_subsample] = ss_sub[:, s_idx:s_idx + 1]

            # pack
            if scf.vectorized_fr:
                if scf.vectorized_early_fr:
                    d = (psi1_f_fr_stacked, spins)
                else:
                    d = (psi1_f_fr_stacked_subdict, spins)
            else:
                d = (psi1_f_frs, spins)
            # note, duplicates were handled in `make_psi1_f_fr_stacked_dict`
            spin_data[spin_down][scale_diff] = d

    return spin_data

# Meta #######################################################################
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
