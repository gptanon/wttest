# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
from ..backend.agnostic_backend import unpad_dyadic
from .scattering1d import scattering1d


def timefrequency_scattering1d(
        x, compute_graph, compute_graph_fr, scattering1d_kwargs,
        backend,
        J_pad, J, log2_T, psi1_f, psi2_f, phi_f, scf,
        pad_mode, pad_left, pad_right,
        ind_start, ind_end, oversampling,
        average, average_global, average_global_phi,
        out_type, out_exclude, paths_exclude, vectorized,
        api_pair_order, do_energy_correction):
    """
    Main function implementing the Joint Time-Frequency Scattering transform.

    For a minimalistic, easier to understand implementation, see
    `examples/jtfs-min/`.

    This code is meant to be read alongside `build_compute_graph_fr` in
    `wavespin/scattering1d/scat_utils_jtfs.py`.

    Below is advanced implementation documentation for developers.

    Frequential scattering
    ======================

    The design paradigm is to orient compute logic around stride, which itself
    serves to meet user specification. Namely, and critically, *padding*
    accomodates *stride*. This JTFS implementation was first done the other way
    around, which tremendously complicated logic and was ultimately flawed.

    Next, design aims to maximize tradeoff between transform size and speed,
    and the information-richness and information-density of resulting coeficients:

        - richness: coefficients result from data, not imputation (padding)
        - density: coefficients pack the most info per sample (non redundancy)

    Refer to `sampling_phi_fr='recalibrate'` in "Compute logic: stride, padding"
    for details.

    Compute logic: stride, padding
    ------------------------------

    Subsampling factors should be decided only by:
       1) input lengths (N_frs)
       2) max permitted subsampling (log2_F if average_fr else J_fr;
                                     assume log2_F throughout this doc)

    Unpad lengths should be decided only by:
       1) input lengths (N_frs)
       2) stride (total_conv_stride_over_U1)
    __________________________________________________________________________

    `out_3D=True` enforces
      - per-`n2` alignment (same `total_conv_stride_over_U1` for all `n1_fr`)
      - per-`n2` unpad length (all same)
    __________________________________________________________________________

    - sub_F_max[n2] = ceil(log2(N_frs[n2]))
    - sub_F_max_all = ceil(log2(N_frs_min))
    - stride_at_max_fr = total_conv_stride_over_U1[n2_max]
    - unpad_len_common_at_max_fr_stride = (ind_end_fr_max[stride_at_max_fr] -
                                           ind_start_fr_max[stride_at_max_fr])
    __________________________________________________________________________

    aligned=True:
        all:
            total_conv_stride_over_U1 = log2_F
            # holds for 'resample' and 'recalibrate' `sampling_psi_fr`,
            # since they only affect `j1_frs`, and there can be no dependence
            # on `n2` or `n1_fr`

        sampling_phi_fr=any:
            out_3D=False:
                - J_pad_frs: minimal to
                    - avoid boundary effects
                    - allow unpadding to at least `unpad_len_fr = 1`

            out_3D=True:
                - J_pad_frs: minimal to
                    - avoid boundary effects, and
                    - allow unpadding to `unpad_len_common_at_max_fr_stride`
                      #1: info-rich alternative to just unpadding maximally,
                      #   or unpadding maximally then zero-repadding.

    aligned=False:
        sampling_phi_fr='resample':
            out_3D=False:
                - total_conv_stride_over_U1 = max_filter_stride

                  max_filter_stride = max(log2_F, j1_frs[n2][n1_fr])
                  #3: stride can vary across `n1_fr` since we don't care for any
                      alignment
                  #4: stride can exceed `log2_F` since conv with `phi_fr` after
                      conv with `psi_fr`, when `j1_fr > log2_F`, cannot reduce
                      subsample-ability

                - J_pad_frs: minimal to avoid boundary effects

            out_3D=True:
                - total_conv_stride_over_U1 = log2_F

                - J_pad_frs: minimal to
                    - avoid boundary effects, and
                    - allow unpadding to `unpad_len_common_at_max_fr_stride`

        sampling_phi_fr='recalibrate':
            # idea with 'recalibrate' and `not aligned` is to maximize
            # info-richness, by
            #  - A: maximizing info-density: max stride
            #  - B: minimizing air-packing: minimizing portion of unpadded signal
            #    that's from padding
            # per A, filter's scale always takes precedence if it *exceeds*
            # competing quantities (i.e. log2_F, sub_F_max)
            #
            # ## Ignore below note, kept for future ##########################
            # 'decimate' with 'recalibrate' is only applied to compensate `j1_fr`
            # to meet `total_conv_stride_over_U1` ############################


            out_3D=False:
                  total_conv_stride_over_U1 = max(min(log2_F, sub_F_max[n2]),
                                                  j1_frs[n2][n1_fr])
                  # `min(log2_F, sub_F_max[n2])` for non-zero unpad length
                  # `max(, j1_frs[n2][n1_fr])` to prevent oversampling.
                  # Former serves B, latter serves A; latter takes priority.

                  max_filter_stride = max(log2_F, j1_frs[n2][n1_fr])
                  #3:
                  #5: stride can exceed `log2_F` since we don't care to enforce
                      a scale of invariance (can skip convolving by `phi_fr`)

                - J_pad_frs: minimal to
                    - avoid boundary effects, and
                    - allow unpadding to at least `unpad_len_fr = 1`

            out_3D=True:
                - total_conv_stride_over_U1 = min_stride_to_unpad_like_max
                  #6: idea is to stride as little as possible to minimize info
                      loss, and minimize number of zero-coeffs from padding,
                      while still allowing to pack 3D (keep `out.shape[-2]` same
                      for all slices); can't do this with
                      `sampling_phi_fr='resample'` since info's lost at `phi_fr`
                  #7: the full criterion is
                          min(min(log2_F, min_stride_to_unpad_like_max),
                              sub_F_max[n2])
                      but
                       - `log2_F < min_stride_to_unpad_like_max` cannot happen,
                         since `N_frs_max` will always subsample by `log2_F`,
                         and all other `N_frs` are smaller and don't need to
                         subsample by more to meet #6, #8
                       - `min_stride_to_unpad_like_max > sub_F_max[n2]` cannot
                         happen, since it'd imply `N_frs_max` unpadded to <1,
                         which is globally forbidden (worst case, F='global',
                         unpads to 1)
                       - can't introduce dependence on `j1_fr` per dependence
                         on `n1_fr`

                  min_stride_to_unpad_like_max = ceil(log2(
                      N_frs[n2] / unpad_len_common_at_max_fr_stride))
                  #8: if we stride less and unpad like max, then we lose info
                      (unpad beyond strided input's length)
                  #9: "unpad_like_max" i.e. `unpad_len_common_at_max_fr_stride`

                - J_pad_frs: minimal to
                    - avoid boundary effects, and
                    - allow unpadding to `unpad_len_common_at_max_fr_stride`

    average_fr_global=True:
        Stride logic in `_joint_lowpass()` is relaxed since all U1 are
        collapsed into one point; will no longer check
        `total_conv_stride_over_U1`

    Note, `sampling_phi_fr`
    -----------------------
    Original design notes read `F_kind='average'` and `F_kind='decimate'` in
    place of `sampling_phi_fr='resample'` and `sampling_phi_fr='recalibrate'`,
    denoting lowpass filtering with Gaussian vs with windowed brickwall for
    decimation, respectively. `F_kind` isn't in the current implementation,
    but `'decimate'` is a superior way to attain the goal of maximizing
    information, at expense of invariance. In that implementation, `F_kind` and
    `sampling_phi_fr` are set independently, but the intended use is `'average'`
    with `'resample'` and `'decimate'` with `'recalibrate'`.  # TODO

    Example scenario
    ----------------
    Suppose `sampling_phi_fr='recalibrate'`, `aligned=False`, `average_fr=True`,
    `log2_F=3`, and `J_fr=5`.

        out_3D=True:
            s = min_stride_to_unpad_like_max
            # can easily drop below log2_F. Suppose s=1.
            # Then, max `n1_fr_subsample` is `1`, and
            # `log2_F_phi_diff=2`, hence `log2_F_phi=1`.

            # If `n1_fr_subsample=1`, then take
            # `phi_f_fr[2][pad_diff][1]`, which yields an
            # unsubsampleable phi (could sub by 2, already did).

            # If `n1_fr_subsample=0`, then take
            # `phi_f_fr[2][pad_diff][0]`, which yields a
            # phi subsampleable by 2, so we attain `s=1`.

        out_3D=False:
            s = max(min(log2_F, N_fr_scale), j1_fr)
            # can easily exceed log2_F. Suppose s=5.
            # Then, max `n1_fr_subsample` is `3`, since
            # the most we can subsample phi by is 8. Take
            # `log2_F_phi_diff=0` and `log2_F_phi=3`,
            # `lowpass_subsample_fr=0`, and then subsample by
            # 4 after phi to attain `s=5`.
            #
            # Suppose N_fr_scale=2, j1_fr=0 -> s=2.
            # Then, `n1_fr_subsample=0`, `log2_F_phi=2`,
            # `log2_F_phi_diff=1`, and `lowpass_subsample_fr=2`.


    Stride, padding: `phi_f` pairs
    ------------------------------
    'phi_f' pairs (psi_t * phi_f, phi_t * phi_f):

    Format:
        configuration:
            total_conv_stride_over_U1
            log2_F_phi
            # comment

    --------------------------------------------------------------------------

    average_fr=True:
        aligned=True:
            log2_F
            log2_F
            # since `total_conv_stride_over_U1 == log2_F` globally

        aligned=False:
            sampling_phi_fr='resample':
                log2_F
                log2_F
                # else we oversample (violate A)

            sampling_phi_fr='recalibrate':
                min(log2_F, max(total_conv_stride_over_U1))
                min(log2_F, max(total_conv_stride_over_U1))
                # we "track" spinned pairs; if `sampling_psi_fr` is 'recalibrate',
                # then `total_conv_stride_over_U1` is lesser with larger
                # `scale_diff`, hence our lowpass is narrower and our features
                # finer. If 'resample', then we accept coarseness in spinned,
                # so we also accept it in phi_f (i.e. the `min` might always
                # evaluate to log2_F).
                # `max` -> phi is no coarser than coarsest spinned
                # (in case of out_3D=False)

    average_fr=False:
        aligned=True:
            0
            log2_F
            # since `total_conv_stride_over_U1 == 0` globally, but we still
            # use same lowpass, and `sampling_phi_fr='recalibrate'` is forbidden

        aligned=False:
            same as `average_fr=True`
            # phi_f is always averaging, so it ignores `average_fr` except for
            # stride, and strides for `aligned=False` are unaffected by
            # `average_fr=False`, except that `out_3D=True` branches are dropped,
            # but that's automatically accounted for in existing expressions

    Combining,

        - `min(log2_F, total_conv_stride_over_U1)` accounts for all of
          `average_fr=True`.
            - `aligned=True` is accounted since the `min` always evaluates to
              `log2_F`. Also `sampling_phi_fr='recalibrate'` is forbidden, but
              even if it wasn't, we'd have `log2_F_phi == log2_F`.
            - `aligned=False`:
                - `sampling_phi_fr='resample': is accounted since if
                  `max(total_conv_stride_over_U1)` exceeds `log2_F`, we take
                  `log2_F`, and if it falls below...
                - `sampling_phi_fr='recalibrate'`: is accounted since
                   the expression exists to account for it.
        - `average_fr=False` see respective comments.

    --------------------------------------------------------------------------
    Complete:

    minmax == min(log2_F, max(total_conv_stride_over_U1))

    if average_fr:
        if aligned:
            total_conv_stride_over_U1_phi = log2_F
            log2_F_phi                    = log2_F
        else:
            if sampling_phi_fr == 'resample':
                total_conv_stride_over_U1_phi = log2_F
                log2_F_phi                    = log2_F
            elif sampling_phi_fr == 'recalibrate':
                total_conv_stride_over_U1_phi = minmax
                log2_F_phi                    = minmax
    else:
        if aligned:
            total_conv_stride_over_U1_phi = 0
            log2_F_phi                    = log2_F
        else:
            if sampling_phi_fr == 'resample':
                total_conv_stride_over_U1_phi = log2_F
                log2_F_phi                    = log2_F
            elif sampling_phi_fr == 'recalibrate':
                total_conv_stride_over_U1_phi = minmax
                log2_F_phi                    = minmax

    --------------------------------------------------------------------------
    Condensed:

    if average_fr or not aligned:
        if aligned or sampling_phi_fr == 'resample':
            total_conv_stride_over_U1_phi = log2_F
        else:
            total_conv_stride_over_U1_phi = minmax
    else:
        total_conv_stride_over_U1_phi = 0

    log2_F_phi = (
        log2_F if (not average_fr and aligned) else
        total_conv_stride_over_U1_phi
    )

    Extended notes: stride, padding
    -------------------------------
    Stride & padding are controlled mainly by `aligned`, `out_3D`, `average_fr`,
    `log2_F`, and `sampling_psi_fr` & `sampling_phi_fr`.

    - `freq` == number of frequential rows (originating from U1), e.g. as in
      `(n1_fr, freq, time)` for joint slice shapes per `n2` (with `out_3D=True`).
    - `n1_fr` == number of frequential joint slices (or their indexing), or
      `psi1_f_fr_*`, per `n2`
    - `n2` == number of `psi2_f` wavelets (or their indexing), together with
      `n1_fr` controlling total number of joint slices

    aligned=True:
        Imposes:
          - `total_conv_stride_over_U1` to be same for all joint coefficients.
            Otherwise, row-to-row log-frequency differences, `dw,2*dw,...`,
            will vary across joint slices, which breaks alignment.
          - `sampling_psi_fr=='resample'`: center frequencies must be same
          - `sampling_phi_fr`: not necessarily restricted, as alignment is
            preserved under different amounts of frequential smoothing, but
            bins may blur together (rather unblur since 'recalibrate' is finer);
            for same fineness/coarseness across slices, 'resample' is required.
            Current implementation forbids `aligned=True` && `'recalibrate'` since
            the use cases aren't sufficient to justify implementation effort.

        average_fr=False:
            Additionally imposes that `total_conv_stride_over_U1==0`,
            since `psi_fr[0]['j'][0] == 0` and `_joint_lowpass()` can no longer
            compensate for variable `j_fr`.

        out_3D=True:
            Additionally imposes that all frequential padding is the same
            (maximum), since otherwise same `total_conv_stride_over_U1`
            yields variable `freq` across `n2`. Though, this isn't strictly
            necessary; see `out_3D` in `help(TimeFrequencyScattering1D)`.

        average_fr_global=True:
            Stride logic in `_joint_lowpass()` is relaxed since all U1 are
            collapsed into one point; will no longer check
            `total_conv_stride_over_U1`

    out_3D=True:
        Imposes:
            - `freq` to be same for all `n2` (can differ due to padding
              or convolutional stride)

    log2_F:
        Larger -> smaller `freq`
        Larger -> greater `max_subsample_before_phi_fr`
        Larger -> greater `J_pad_frs_max`
                  (unless `J_fr` & `Q_fr` yield greater padding)

    Debug tips
    ----------
      - Check the following and related scf attributes:
          - N_frs, N_fr_scales
          - J_pad_frs, J_pad_frs_max_init
          - ind_end_fr, ind_end_fr_max
    """
    # pack for later
    B = backend
    if out_exclude is None:
        out_exclude = []
    N = x.shape[-1]
    commons = (B, scf, do_energy_correction, compute_graph_fr, out_exclude,
               paths_exclude, average, average_global, phi_f)
    Dearly = compute_graph_fr['Dearly']

    out_S_0 = []
    out_S_1_tm = []
    out_S_1 = {'phi_t * phi_f': []}
    out_S_2 = {'psi_t * psi_f': {1: [], -1: []},
               'psi_t * phi_f': [],
               'phi_t * psi_f': [[]]}

    # Time scattering ########################################################
    (jtfs_cfg_in_s1d, include_phi_t, do_joint_complex, skip_spinned) = [
        Dearly[k] for k in
        ('jtfs_cfg_in_s1d', 'include_phi_t', 'do_joint_complex', 'skip_spinned')
    ]

    # do the scattering
    out_tm = scattering1d(x, **scattering1d_kwargs, jtfs_cfg=jtfs_cfg_in_s1d)

    # unpack
    if jtfs_cfg_in_s1d['do_S_0']:
        S_0 = out_tm['S_0']
    if jtfs_cfg_in_s1d['do_S_1_tm']:
        S_1_tms = out_tm['S_1_tms']
    if jtfs_cfg_in_s1d['do_S_1_avg']:
        S_1_avgs = out_tm['S_1_avgs']
    if jtfs_cfg_in_s1d['jtfs_needs_U_2_c']:
        U_2_cs = out_tm['U_2_cs']

    # Finish first-order time scattering #####################################
    U_1_dict, U_12_dict, keys1_grouped_inverse = [
        compute_graph[k] for k in
        ('U_1_dict', 'U_12_dict', 'keys1_grouped_inverse')
    ]

    # S0: execute
    if 'S0' not in out_exclude:
        D = Dearly['S0']
        if do_energy_correction and average:
            S_0 = _energy_correction(S_0, D['energy_correction'])
        out_S_0.append({'coef': S_0,
                        'j': (log2_T,) if average else (),
                        'n': (-1,)     if average else (),
                        'spin': (),
                        'stride': (D['k0'],) if average else ()})

    # S1: execute
    if 'S1' not in out_exclude:
        # energy correction due to stride & inexact unpad length
        De_S1 = Dearly['S1']
        do_scaling_S1 = De_S1['do_scaling']
        if do_scaling_S1:
            S1_ec = De_S1['energy_correction']

        if vectorized and average:  # TODO if list instead? & other places
            S_1_tms = B.concatenate(S_1_tms)
            if do_scaling_S1:
                S_1_tms = _energy_correction(S_1_tms, S1_ec)
            S_1_avgs = S_1_tms
        else:
            if do_scaling_S1:
                for n1, k1 in keys1_grouped_inverse.items():
                    S_1_tms[n1] = _energy_correction(S_1_tms[n1], S1_ec[k1])

        # append into outputs ############################################
        for n1, k1 in keys1_grouped_inverse.items():
            j1 = psi1_f[n1]['j']
            coef = (S_1_tms[:, n1:n1+1] if vectorized and average else
                    S_1_tms[n1])
            out_S_1_tm.append({'coef': coef,
                               'j': (j1,), 'n': (n1,), 'spin': (),
                               'stride': (De_S1['total_conv_stride_tms'][k1],)})

        # for backends that won't update in-place in `_energy_correction`
        # (e.g. tensorflow)
        if average:
            S_1_avgs = S_1_tms

    # `phi_t *`: append for further processing
    if include_phi_t:
        # energy correction, if not done
        S_1_avg_energy_corrected = bool(average and
                                        'S1' not in out_exclude)
        if not S_1_avg_energy_corrected:
            do_scaling_phi_t = Dearly['phi_t']['do_scaling']
            if do_scaling_phi_t:
                phi_t_ec = Dearly['phi_t']['energy_correction']

            if vectorized:
                S_1_avgs = B.concatenate(S_1_avgs)
                if do_scaling_phi_t:
                    S_1_avgs = _energy_correction(S_1_avgs, phi_t_ec)
            else:
                if do_scaling_phi_t:
                    for n1, k1 in keys1_grouped_inverse.items():
                        S_1_avgs[n1] = _energy_correction(S_1_avgs[n1], phi_t_ec)

    # Frequential averaging over time averaged coefficients ##################
    # `U1 * (phi_t * phi_f)` pair
    if include_phi_t:
        # zero-pad along frequency
        pad_fr = scf.J_pad_frs_max
        n_n1s = (S_1_avgs.shape[1] if vectorized else
                 len(S_1_avgs))
        if n_n1s < 2**pad_fr:
            # usual case
            S_1_avg = _right_pad(S_1_avgs, pad_fr, scf, B)
        else:
            # handle rare edge case (see `_J_pad_fr_fo` docs)
            assert (scf.max_pad_factor_fr is not None and
                    max(scf.max_pad_factor_fr[0] == 0)), scf.max_pad_factor_fr
            if vectorized and average:
                S_1_avg = S_1_avgs[:, :2**pad_fr]
            else:
                S_1_avg = S_1_avgs[:2**pad_fr]

        if (('phi_t * phi_f' not in out_exclude and
             not scf.average_fr_global_phi) or
                'phi_t * psi_f' not in out_exclude):
            # map frequency axis to Fourier domain
            S_1_avg_hat = B.r_fft(S_1_avg, axis=-2)

    if 'phi_t * phi_f' not in out_exclude:
        D = Dearly['phi_t * phi_f']
        (lowpass_subsample_fr, n1_fr_subsample, log2_F_phi) = [
            D[k] for k in
            ('lowpass_subsample_fr', 'n1_fr_subsample', 'log2_F_phi')
        ]

        if scf.average_fr_global_phi:
            # take mean along frequency directly
            S_1 = B.mean(S_1_avg, axis=-2)
        else:
            # Low-pass filtering over frequency
            phi_fr = scf.phi_f_fr[D['log2_F_phi_diff']][D['pad_diff']][0]
            S_1_c = B.multiply(S_1_avg_hat, phi_fr)
            S_1_hat = B.subsample_fourier(S_1_c, 2**n1_fr_subsample, axis=-2)
            S_1_r = B.ifft_r(S_1_hat, axis=-2)

        # Unpad frequency
        if not scf.average_fr_global_phi:
            S_1 = B.unpad(S_1_r, D['ind_start_fr'], D['ind_end_fr'], axis=-2)

        # energy correction due to stride & inexact unpad indices;
        # or, global averaging pad factor correction
        # time already done
        # TODO rename energy_correction to scaling_factor?
        if D['do_scaling']:
            S_1 = _energy_correction(S_1, D['energy_correction'])

        scf.__total_conv_stride_over_U1 = D['total_conv_stride_over_U1_realized']
        # append to out with meta
        out_S_1['phi_t * phi_f'].append({
            'coef': S_1, 'j': (log2_T, log2_F_phi), 'n': (-1, -1),
            'spin': (0,), 'stride': D['stride']})
    else:
        scf.__total_conv_stride_over_U1 = -1

    ##########################################################################
    # Joint scattering: separable convolutions (along time & freq), and low-pass
    # `U1 * (psi_t * psi_f)` (up & down), and `U1 * (psi_t * phi_f)`
    if do_joint_complex:
        DT, DF, DL = [compute_graph_fr[k] for k in ('DT', 'DF', 'DL')]
        for n2 in U_12_dict:
            (Dn2_all, DTn2, DFn2, DFn2_n1_frs, DFn2_n1_fr_subsamples,
             DLn2_n1_frs, DLn2_n1_fr_subsamples) = _unpack_compute_graph_n2(
                 n2, DT, DF, DL)

            j2 = psi2_f[n2]['j']
            # fetch the complex-valued second-order time-scattered coefficients
            Y_2 = U_2_cs[n2]

            # TODO rm?
            n_n1s = (Y_2.shape[1] if vectorized else
                     len(Y_2))
            assert n_n1s == scf.N_frs[n2]

            # Unpad early if possible
            Y_2, trim_tm = _maybe_unpad_time(Y_2, DTn2, B)

            # temporal pad modification
            k1_plus_k2 = DTn2['k1_plus_k2']
            if pad_mode == 'reflect' and average:
                if isinstance(Y_2, list):
                    Y_2 = B.concatenate(Y_2, axis=-2)
                # `=` since tensorflow makes copy
                Y_2 = B.conj_reflections(
                    Y_2, ind_start, ind_end,
                    k1_plus_k2, N, J_pad, log2_T, J,
                    pad_left, pad_right, trim_tm)

            # frequential pad
            pad_fr_to_execute = DTn2['pad_fr_to_execute']
            if scf.pad_mode_fr == 'custom':
                Y_2_arr = scf.pad_fn_fr(Y_2, pad_fr_to_execute, scf, B)
            else:
                Y_2_arr = _right_pad(Y_2, pad_fr_to_execute, scf, B)

            # map to Fourier along (log-)frequency to prepare for conv,
            # and handle special case
            Y_2_hat = B.fft(Y_2_arr, axis=-2)
            if DTn2['separate_pad_psi_phi']:
                Y_2_arr_psi = B.unpad(Y_2_arr, DTn2['unpad_larger_start'],
                                      DTn2['unpad_larger_end'], axis=-2)
                Y_2_hat_psi = B.fft(Y_2_arr_psi, axis=-2)
            else:
                Y_2_hat_psi = Y_2_hat

            # Transform over frequency + low-pass, for both spins ############
            # `* psi_f` part of `U1 * (psi_t * psi_f)`
            if not skip_spinned:
                _frequency_scattering(Y_2_hat_psi, Dn2_all, j2, n2,
                                      k1_plus_k2, trim_tm, commons,
                                      out_S_2['psi_t * psi_f'])

            # Low-pass over frequency ########################################
            # `* phi_f` part of `U1 * (psi_t * phi_f)`
            if 'psi_t * phi_f' not in out_exclude:
                _frequency_lowpass(Y_2_hat, Y_2_arr, Dn2_all, j2, n2,
                                   k1_plus_k2, trim_tm, commons,
                                   out_S_2['psi_t * phi_f'])

    ##########################################################################
    # `U1 * (phi_t * psi_f)`
    if 'phi_t * psi_f' not in out_exclude:
        # unpack compute graph
        (Dn2_all, DTn2, DFn2, DFn2_n1_frs, DFn2_n1_fr_subsamples,
         DLn2_n1_frs, DLn2_n1_fr_subsamples) = _unpack_compute_graph_n2(
             -1, DT, DF, DL)

        # take largest subsampling factor
        j2 = log2_T
        k1_plus_k2 = (max(log2_T - oversampling, 0) if not average_global_phi else
                      log2_T)
        # n2_time = U_0.shape[-1] // 2**max(j2 - oversampling, 0)

        # reuse from first-order scattering
        Y_2_hat = S_1_avg_hat

        # Transform over frequency + low-pass
        # `* psi_f` part of `U1 * (phi_t * psi_f)`
        _frequency_scattering(Y_2_hat, Dn2_all, j2, -1, k1_plus_k2, 0,
                              commons, out_S_2['phi_t * psi_f'], for_phi_f=True)

    ##########################################################################
    # pack outputs & return
    out = {}
    out['S0'] = out_S_0
    out['S1'] = out_S_1_tm
    out['phi_t * phi_f'] = out_S_1['phi_t * phi_f']
    out['phi_t * psi_f'] = out_S_2['phi_t * psi_f'][0]
    out['psi_t * phi_f'] = out_S_2['psi_t * phi_f']
    out['psi_t * psi_f_up'] = out_S_2['psi_t * psi_f'][1]
    out['psi_t * psi_f_dn'] = out_S_2['psi_t * psi_f'][-1]
    assert tuple(out) == api_pair_order, (tuple(out), api_pair_order)

    # delete excluded
    for pair in out_exclude:
        del out[pair]

    # warn of any zero-sized coefficients
    for pair in out:
        for i, c in enumerate(out[pair]):
            if 0 in c['coef'].shape:  # no-cov
                import warnings
                warnings.warn("out[{}][{}].shape == {}".format(
                    pair, i, c['coef'].shape))

    # concat
    if out_type == 'dict:array':
        # handle S0, S1
        for k in ('S0', 'S1'):
            if k in out:
                out[k] = B.concatenate([c['coef'] for c in out[k]], axis=1)
        # handle joint
        for k, v in out.items():
            if k not in ('S0', 'S1'):
                if scf.out_3D:
                    # stack joint slices, preserve 3D structure
                    out[k] = B.concatenate([c['coef'] for c in v], axis=1,
                                           keep_cat_dim=True)
                else:
                    # flatten joint slices, return 2D
                    try:
                        out[k] = B.concatenate([c['coef'] for c in v], axis=1)
                    except:
                        1/0

    elif out_type == 'dict:list':
        pass  # already done

    else:
        if scf.out_3D:
            # cannot concatenate `S0` & `S1` with joint slices, return separately,
            # even if 'list' for consistency
            o0 = [c for k, v in out.items() for c in v
                  if k in ('S0', 'S1')]
            o1 = [c for k, v in out.items() for c in v
                  if k not in ('S0', 'S1')]
        else:
            # flatten all to then concat along `freq` dim
            o = [c for v in out.values() for c in v]

        if out_type == 'array':
            if scf.out_3D:
                out_0 = B.concatenate([c['coef'] for c in o0], axis=1)
                out_1 = B.concatenate([c['coef'] for c in o1], axis=1,
                                      keep_cat_dim=True)
                out = (out_0, out_1)
            else:
                out = B.concatenate([c['coef'] for c in o], axis=1)
        elif out_type == 'list':
            if scf.out_3D:
                out = (o0, o1)
            else:
                out = o

    return out


def _frequency_scattering(Y_2_hat, Dn2_all, j2, n2, k1_plus_k2, trim_tm,
                          commons, out_S_2, for_phi_f=False):
    # unpack params & compute graph ------------------------------------------
    B, scf, _, compute_graph_fr, out_exclude, paths_exclude, *_ = commons
    (DTn2, DFn2, DFn2_n1_frs, DFn2_n1_fr_subsamples, DLn2_n1_frs,
     DLn2_n1_fr_subsamples) = Dn2_all

    unpad_early_fr = DFn2['unpad_early_fr']
    # ------------------------------------------------------------------------

    scale_diff = scf.scale_diffs[n2]
    psi_id = scf.psi_ids[scale_diff]
    pad_fr = DTn2['pad_fr_psi']
    pad_diff = scf.J_pad_frs_max_init - pad_fr

    # Transform over frequency + low-pass, for both spins (if `spin_down`)
    if not scf.vectorized_fr:
        psi1_f_frs, spins = compute_graph_fr['spin_data'][for_phi_f][scale_diff]

        for spin, psi1_f_fr in zip(spins, psi1_f_frs):
            for n1_fr in range(len(psi1_f_fr)):
                if n1_fr in paths_exclude.get('n1_fr', {}):
                    continue

                # fetch params
                j1_fr = scf.psi1_f_fr_up['j'][psi_id][n1_fr]
                log2_F_phi_diff, n1_fr_subsample = [
                    DFn2_n1_frs[n1_fr][k] for k in
                    ('log2_F_phi_diff', 'n1_fr_subsample')]

                # Wavelet transform over frequency
                Y_1_fr_c = B.multiply(Y_2_hat, psi1_f_fr[n1_fr])
                Y_1_fr_hat = B.subsample_fourier(Y_1_fr_c, 2**n1_fr_subsample,
                                                 axis=-2)
                Y_1_fr_c = B.ifft(Y_1_fr_hat, axis=-2)

                # Unpad early if possible
                D = DLn2_n1_frs[n1_fr]
                if unpad_early_fr:
                    Y_1_fr_c = B.unpad(Y_1_fr_c, D['ind_start_fr'],
                                       D['ind_end_fr'], axis=-2)

                # Modulus
                U_2_m = B.modulus(Y_1_fr_c)

                # Convolve by Phi = phi_t * phi_f, unpad
                S_2, stride = _joint_lowpass(
                    U_2_m, D, n2, n1_fr, n1_fr_subsample, log2_F_phi_diff,
                    k1_plus_k2, trim_tm, pad_diff, commons)

                # append to out
                out_S_2[spin].append(
                    {'coef': S_2, 'j': (j2, j1_fr), 'n': (n2, n1_fr),
                     'spin': (spin,), 'stride': stride})
    else:
        # execute compute graph ##############################################
        Y_1_fr_dict = compute_graph_fr['Y_1_fr_dict'][scale_diff]

        # Wavelet transform over frequency -----------------------------------
        # First handle multiplication
        # `(batch_size, n_n1s, t) == Y_2_hat.shape`, so broadcast to enable
        #     (batch_size, 1,    1,        n_n1s, t) *
        #     (1,          spin, n_n1_frs, n_n1s, t)`
        Y_1_fr_cs = {}
        if scf.vectorized_early_fr:
            psi1_f_fr_stacked, spins = compute_graph_fr[
                'spin_data'][for_phi_f][scale_diff]
            _Y_1_fr_cs = B.multiply(Y_2_hat[:, None, None], psi1_f_fr_stacked)

            # group by `n1_fr_subsample` for subsequent processing
            start = 0
            for n1_fr_subsample in Y_1_fr_dict:
                n_n1_frs = len(Y_1_fr_dict[n1_fr_subsample])
                end = start + n_n1_frs
                Y_1_fr_cs[n1_fr_subsample] = _Y_1_fr_cs[:, :, start:end]
                start = end

        else:
            psi1_f_fr_stacked_subdict, spins = compute_graph_fr[
                'spin_data'][for_phi_f][scale_diff]

            for n1_fr_subsample in Y_1_fr_dict:
                Y_1_fr_cs[n1_fr_subsample] = B.multiply(
                    Y_2_hat[:, None, None],
                    psi1_f_fr_stacked_subdict[n1_fr_subsample])

        # Now subsampling & ifft
        Y_1_fr_cs_grouped = {}
        for n1_fr_subsample, Y_1_fr_c in Y_1_fr_cs.items():
            Y_1_fr_hat = B.subsample_fourier(Y_1_fr_c, 2**n1_fr_subsample,
                                             axis=-2)
            Y_1_fr_c = B.ifft(Y_1_fr_hat, axis=-2)

            # Unpad early if possible
            if unpad_early_fr:
                _D = DLn2_n1_fr_subsamples[n1_fr_subsample]
                Y_1_fr_c = B.unpad(
                    Y_1_fr_c, _D['ind_start_fr'], _D['ind_end_fr'], axis=-2)

            # Group by subsampling factor
            Y_1_fr_cs_grouped[n1_fr_subsample] = Y_1_fr_c

        # Joint lowpass ------------------------------------------------------
        # Convolve by Phi = phi_t * phi_f
        # The input to the method must be a tensor. Time-first lowpassing
        # enables concatenating along frequency.
        # `(batch_size, spin, n_n1_frs, n_n1s, t) == Y_1_fr_c.shape`, so
        # merge `n_n1_frs` and `n_n1s` to account for both being variable
        Y_1_fr_c = []
        # free memory as we append, and store shapes
        shapes_orig = []
        n_unrolleds = []
        for n1_fr_subsample in Y_1_fr_dict:
            c = Y_1_fr_cs_grouped.pop(n1_fr_subsample)
            s = list(c.shape)
            n_unrolled = s[2] * s[3]  # TODO graph?
            shape = [s[0], s[1], n_unrolled, s[4]]
            Y_1_fr_c.append(B.reshape(c, shape))

            shapes_orig.append(s)
            n_unrolleds.append(n_unrolled)
        # cat
        Y_1_fr_c = B.concatenate(Y_1_fr_c, axis=-2)
        # Modulus
        U_2_m = B.modulus(Y_1_fr_c)
        # Convolve by `phi_t`, unpad
        # relevant params are same for all `n1_fr`
        D = list(DLn2_n1_frs.values())[0]
        S_2_r = _joint_lowpass_part1(U_2_m, D, n2, None, k1_plus_k2, trim_tm,
                                     commons)

        # unpack the result based on whether further compute can group
        # by `n1_fr_subsample` (always True for `oversampling_fr = 0`)
        group_by_n1_fr_subsample = DFn2['part2_grouped_by_n1_fr_subsample']
        n_t = S_2_r.shape[-1]
        S_2_r_grouped = {}
        if group_by_n1_fr_subsample:  # TODO graph?
            # unpack into grouped `n1_fr`-slices
            start = 0
            for n, s, n1_fr_subsample in zip(
                    n_unrolleds, shapes_orig, Y_1_fr_dict):
                # undo reshaping
                s[-1] = n_t

                # `(batch_size, spin, n_n1_frs * n_n1s, t) == S_2_r.shape`
                end = start + n
                S_2_r_grouped[n1_fr_subsample] = B.reshape(
                    S_2_r[:, :, start:end], s)
                start = end
        else:
            # Unpack into individual `n1_fr`s.
            # for each `n1_fr_subsample`
            #    - we have some `n1_fr`s.
            #    - to unpack each `n1_fr`, we use their length, `n_n1s`.
            #    - the start index of each `n1_fr` in the *unrolled* tensor
            #      is incremented by both, the current `n1_fr_subsample`, which
            #      is incremented by the total increment of the previously
            #      unpacked `n1_fr_subsample`, and previously unpacked `n1_fr`s
            #      in the current `n1_fr_subsample`.
            n1_fr_subsample_unrolled_start = 0
            for n, s, n1_fr_subsample in zip(
                    n_unrolleds, shapes_orig, Y_1_fr_dict):
                n_n1s = DFn2_n1_fr_subsamples[n1_fr_subsample]['n_n1s_pre_part2']
                # 1 `n1_fr` slice, `n_n1s` n1s, n_t times;
                # preserve batch dim, s[0], and number of spins, s[1]
                s[2:] = (1, n_n1s, n_t)

                n1_fr_subsample_unrolled_end = n1_fr_subsample_unrolled_start + n

                n1_fr_start = 0
                for n1_fr in Y_1_fr_dict[n1_fr_subsample]:
                    n1_fr_end = n1_fr_start + n_n1s

                    n1_start = n1_fr_subsample_unrolled_start + n1_fr_start
                    n1_end = n1_start + n_n1s
                    S_2_r_grouped[n1_fr] = B.reshape(
                        S_2_r[:, :, n1_start:n1_end], s)
                    n1_fr_start = n1_fr_end
                assert n1_end == n1_fr_subsample_unrolled_end
                n1_fr_subsample_unrolled_start = n1_fr_subsample_unrolled_end

        # Convolve by `phi_f`, unpad, append to output
        commons2 = (scf, Y_1_fr_dict, psi_id, spins, j2, out_S_2,
                    group_by_n1_fr_subsample)
        if group_by_n1_fr_subsample:
            for n1_fr_subsample in Y_1_fr_dict:
                _DL = DLn2_n1_fr_subsamples[n1_fr_subsample]
                _DF = DFn2_n1_fr_subsamples[n1_fr_subsample]
                log2_F_phi_diff = _DF['log2_F_phi_diff']

                _do_part2_and_append_output(
                    S_2_r_grouped.pop(n1_fr_subsample),
                    _DL, n2, None, n1_fr_subsample, pad_diff, log2_F_phi_diff,
                    commons, *commons2)
        else:
            n1_frs = list(S_2_r_grouped)
            for n1_fr in n1_frs:
                _DL = DLn2_n1_frs[n1_fr]
                _DF = DFn2_n1_frs[n1_fr]
                log2_F_phi_diff = _DF['log2_F_phi_diff']
                n1_fr_subsample = _DF['n1_fr_subsample']

                _do_part2_and_append_output(
                    S_2_r_grouped.pop(n1_fr),
                    _DL, n2, n1_fr, n1_fr_subsample, pad_diff, log2_F_phi_diff,
                    commons, *commons2)


def _do_part2_and_append_output(S_2_r, _DL, n2, n1_fr, n1_fr_subsample, pad_diff,
                                log2_F_phi_diff, commons, scf, Y_1_fr_dict,
                                psi_id, spins, j2, out_S_2,
                                group_by_n1_fr_subsample):
    S_2, stride = _joint_lowpass_part2(
        S_2_r, _DL, n2, n1_fr, n1_fr_subsample, pad_diff, log2_F_phi_diff,
        commons)

    # append to output
    for s1_fr, spin in enumerate(spins):
        if len(spins) == 2:
            s_idx = (1, 0)[s1_fr]
        else:
            s_idx = 0  # the only possible option
        # see "SPIN NOTE" in `scat_utils.py`  # TODO
        psi1_f_fr = {1:  scf.psi1_f_fr_dn,
                     0:  scf.psi1_f_fr_dn,  # phi_t * psi_f
                     -1: scf.psi1_f_fr_up}[spin]

        if group_by_n1_fr_subsample:
            # TODO rm
            a, b = S_2.shape[2], len(Y_1_fr_dict[n1_fr_subsample])
            assert a == b, (a, b)

            for idx, n1_fr in enumerate(Y_1_fr_dict[n1_fr_subsample]):
                j1_fr = psi1_f_fr['j'][psi_id][n1_fr]
                coef = S_2[:, s_idx, idx]
                assert coef.ndim == 3  # TODO rm
                out_S_2[spin].append(
                    {'coef': coef, 'j': (j2, j1_fr), 'n': (n2, n1_fr),
                     'spin': (spin,), 'stride': stride})
        else:  # TODO only do subsampling if lowpass_fr != 0 ?
            j1_fr = psi1_f_fr['j'][psi_id][n1_fr]
            coef = S_2[:, s_idx, 0]
            assert coef.ndim == 3  # TODO rm
            out_S_2[spin].append(
                {'coef': coef, 'j': (j2, j1_fr), 'n': (n2, n1_fr),
                 'spin': (spin,), 'stride': stride})


def _frequency_lowpass(Y_2_hat, Y_2_arr, Dn2_all, j2, n2, k1_plus_k2,
                       trim_tm, commons, out_S_2):
    # unpack params & compute graph ------------------------------------------
    B, scf, *_ = commons
    DTn2, _, DFn2_n1_frs, _, DLn2_n1_frs, _ = Dn2_all
    D = DLn2_n1_frs[-1]

    (total_conv_stride_over_U1_phi, log2_F_phi, log2_F_phi_diff,
     n1_fr_subsample, unpad_early_fr) = [
         DFn2_n1_frs[-1][k] for k in
         ('total_conv_stride_over_U1', 'log2_F_phi', 'log2_F_phi_diff',
          'n1_fr_subsample', 'unpad_early_fr')
    ]
    # ------------------------------------------------------------------------

    pad_fr = DTn2['pad_fr_phi']
    pad_diff = scf.J_pad_frs_max_init - pad_fr

    # lowpassing
    # TODO why `Y_1_*`?
    if scf.average_fr_global_phi:
        Y_1_fr_c = B.mean(Y_2_arr, axis=-2)
    else:
        # convolve
        Y_1_fr_c = B.multiply(Y_2_hat, scf.phi_f_fr[log2_F_phi_diff][pad_diff][0])
        Y_fr_hat = B.subsample_fourier(Y_1_fr_c, 2**n1_fr_subsample, axis=-2)
        Y_1_fr_c = B.ifft(Y_fr_hat, axis=-2)

    # if n2 == 4:  # TODO
    #     1/0

    # maybe unpad
    if unpad_early_fr:
        Y_1_fr_c = B.unpad(Y_1_fr_c, D['ind_start_fr'], D['ind_end_fr'], axis=-2)

    # Modulus
    U_2_m = B.modulus(Y_1_fr_c)

    # Convolve by Phi = phi_t * phi_f
    S_2, stride = _joint_lowpass(
        U_2_m, D, n2, -1, n1_fr_subsample, log2_F_phi_diff,
        k1_plus_k2, trim_tm, pad_diff, commons)

    out_S_2.append({'coef': S_2, 'j': (j2, log2_F_phi), 'n': (n2, -1),
                    'spin': (0,), 'stride': stride})


def _joint_lowpass(U_2_m, D, n2, n1_fr, n1_fr_subsample, log2_F_phi_diff,
                   k1_plus_k2, trim_tm, pad_diff, commons):
    S_2_r = _joint_lowpass_part1(U_2_m, D, n2, n1_fr, k1_plus_k2, trim_tm,
                                 commons)
    S_2, stride = _joint_lowpass_part2(S_2_r, D, n2, n1_fr, n1_fr_subsample,
                                       pad_diff, log2_F_phi_diff, commons)
    return S_2, stride


def _joint_lowpass_part1(U_2_m, D, n2, n1_fr, k1_plus_k2, trim_tm, commons):
    B, scf, _, _, _, _, average, average_global, phi_f = commons

    # time lowpassing ########################################################
    if D['do_averaging']:
        if average_global:
            S_2_r = B.mean(U_2_m, axis=-1)
        elif average:
            # Low-pass filtering over time
            U_2_hat = B.r_fft(U_2_m)
            S_2_c = B.multiply(U_2_hat, phi_f[trim_tm][k1_plus_k2])
            S_2_hat = B.subsample_fourier(S_2_c, 2**D['k2_log2_T'])
            S_2_r = B.ifft_r(S_2_hat)
    else:
        S_2_r = U_2_m

    # `not average` and `n2 == -1` already unpadded
    if D['do_unpad_tm']:
        S_2_r = B.unpad(S_2_r, D['ind_start_tm'], D['ind_end_tm'])

    return S_2_r


def _joint_lowpass_part2(
        S_2_r, D, n2, n1_fr, n1_fr_subsample, pad_diff, log2_F_phi_diff, commons):
    B, scf, do_energy_correction, *_ = commons

    # freq lowpassing ########################################################
    if D['do_averaging_fr']:
        if scf.average_fr_global:
            S_2_fr = B.mean(S_2_r, axis=-2)
        elif scf.average_fr:
            if scf.F_kind == 'average':
                # Low-pass filtering over frequency
                phi_fr = scf.phi_f_fr[log2_F_phi_diff][pad_diff][n1_fr_subsample]
                U_2_hat = B.r_fft(S_2_r, axis=-2)
                S_2_fr_c = B.multiply(U_2_hat, phi_fr)
                S_2_fr_hat = B.subsample_fourier(
                    S_2_fr_c, 2**D['lowpass_subsample_fr'], axis=-2)
                S_2_fr = B.ifft_r(S_2_fr_hat, axis=-2)
            elif scf.F_kind == 'decimate':
                assert scf.oversampling_fr == 0  # future todo
                if D['lowpass_subsample_fr'] != 0:
                    S_2_fr = scf.decimate(
                        S_2_r, 2**D['lowpass_subsample_fr'], axis=-2)
                else:
                    S_2_fr = S_2_r
    else:
        S_2_fr = S_2_r

    # fr unpad/repad
    if D['do_repad_fr']:
        # first do conventional unpadding w.r.t. `N_fr` to drop boundary effects
        if D['do_unpad_fr']:
            S_2_fr = B.unpad(S_2_fr, D['ind_start_N_fr'], D['ind_end_N_fr'],
                             axis=-2)
        S_2_fr = _pad_zero(S_2_fr, D['ind_end_fr'] - D['ind_start_fr'], B)
    elif D['do_unpad_fr']:
        S_2_fr = B.unpad(S_2_fr, D['ind_start_fr'], D['ind_end_fr'], axis=-2)

    S_2 = S_2_fr

    # energy correction ######################################################
    # correction due to stride & inexact unpad indices;
    # or, global averaging correction
    if D['do_scaling']:
        S_2 = _energy_correction(S_2, D['energy_correction'])

    # sanity checks (see "Subsampling, padding") #############################
    if scf.aligned:
        if not D['global_averaged_fr']:
            # `total_conv_stride_over_U1` renamed; comment for searchability
            if scf.__total_conv_stride_over_U1 == -1:
                scf.__total_conv_stride_over_U1 = (  # set if not set yet
                    D['total_conv_stride_over_U1_realized'])
            if not scf.average_fr:
                assert D['total_conv_stride_over_U1_realized'] == 0
            else:
                a = D['total_conv_stride_over_U1_realized']
                b = scf.__total_conv_stride_over_U1
                assert a == b, (a, b)

    # ensure we didn't lose info (over-trim)
    unpad_len_fr = S_2.shape[-2]
    assert (unpad_len_fr >=
            scf.N_frs[n2] / 2**D['total_conv_stride_over_U1_realized']
            ), (unpad_len_fr, scf.N_frs, D['total_conv_stride_over_U1_realized'])
    # ensure we match specified unpad len (possible to fail by under-padding)
    if D['do_unpad_fr']:
        unpad_len_fr_expected = D['ind_end_fr'] - D['ind_start_fr']
        assert unpad_len_fr_expected == unpad_len_fr, (
            unpad_len_fr_expected, unpad_len_fr)

    stride = (D['total_conv_stride_over_U1_realized'], D['total_conv_stride_tm'])
    return S_2, stride


#### helper methods ##########################################################
def _unpack_compute_graph_n2(n2, DT, DF, DL):
    DTn2, DFn2, _DLn2 = [_D[n2] for _D in (DT, DF, DL)]
    DFn2_n1_frs = DFn2['g:n1_frs']
    DFn2_n1_fr_subsamples = DFn2['g:n1_fr_subsamples']
    DLn2_n1_frs = _DLn2['g:n1_frs']
    DLn2_n1_fr_subsamples = _DLn2['g:n1_fr_subsamples']
    Dn2_all = (DTn2, DFn2, DFn2_n1_frs, DFn2_n1_fr_subsamples,
               DLn2_n1_frs, DLn2_n1_fr_subsamples)
    return (Dn2_all, DTn2, DFn2, DFn2_n1_frs, DFn2_n1_fr_subsamples,
            DLn2_n1_frs, DLn2_n1_fr_subsamples)


def _right_pad(coeffs, pad_fr, scf, B):
    if scf.pad_mode_fr == 'conj-reflect-zero':
        return _pad_conj_reflect_zero(coeffs, 2**pad_fr, scf.N_frs_max_all, B)
    return _pad_zero(coeffs, 2**pad_fr, B)


def _pad_zero(coeffs, padded_len, B):
    # zero-pad
    if isinstance(coeffs, list):
        zero_row = B.zeros_like(coeffs[0])
        zero_rows = [zero_row] * (padded_len - len(coeffs))
        out = B.concatenate(coeffs + zero_rows, axis=-2)
    else:
        s = list(coeffs.shape)
        s[-2] = padded_len
        out = B.zeros_like(coeffs, shape=s)
        coeffs_len = int(coeffs.shape[-2])
        if B.name not in ('tensorflow', 'jax'):
            out[..., :coeffs_len, :] = coeffs
        else:
            out = B.assign_slice(out, coeffs, slice(0, coeffs_len), axis=-2)
    return out


def _pad_conj_reflect_zero(coeffs, padded_len, N_frs_max_all, B):
    if not isinstance(coeffs, list):
        # preallocating the output array won't make much performance difference
        # here, so always use list
        coeffs = [coeffs[..., i:i+1, :] for i in range(coeffs.shape[1])]

    n_coeffs_input = len(coeffs)  # == N_fr
    zero_row = B.zeros_like(coeffs[0])
    # first zero pad, then reflect remainder (including zeros as appropriate)
    n_zeros = min(N_frs_max_all - n_coeffs_input,  # never need more than this
                  padded_len - n_coeffs_input)     # cannot exceed `padded_len`
    zero_rows = [zero_row] * n_zeros

    coeffs_new = coeffs + zero_rows
    right_pad = max((padded_len - n_coeffs_input) // 2, n_zeros)
    left_pad  = padded_len - right_pad - n_coeffs_input

    # right pad
    right_rows = zero_rows
    idx = -2
    reflect = False
    while len(right_rows) < right_pad:
        c = coeffs_new[idx]
        c = c if reflect else B.conj(c)
        right_rows.append(c)
        if idx in (-1, -len(coeffs_new)):
            reflect = not reflect
        idx += 1 if reflect else -1

    # (circ-)left pad
    left_rows = []
    idx = - (len(coeffs_new) - 1)
    reflect = False
    while len(left_rows) < left_pad:
        c = coeffs_new[idx]
        c = c if reflect else B.conj(c)
        left_rows.append(c)
        if idx in (-1, -len(coeffs_new)):
            reflect = not reflect
        idx += -1 if reflect else 1
    left_rows = left_rows[::-1]

    return B.concatenate(coeffs + right_rows + left_rows, axis=-2)


def _maybe_unpad_time(Y_2_c, DTn2, B):
    # handle `vectorized`
    if isinstance(Y_2_c, list):
        for i in range(len(Y_2_c)):
            Y_2_c[i], trim_tm = __maybe_unpad_time(Y_2_c[i], DTn2, B)
    else:
        Y_2_c, trim_tm = __maybe_unpad_time(Y_2_c, DTn2, B)
    return Y_2_c, trim_tm


def __maybe_unpad_time(Y_2_c, DTn2, B):
    if DTn2['do_unpad_dyadic']:
        Y_2_c = unpad_dyadic(Y_2_c, **DTn2['unpad_dyadic_kwargs'])
    elif DTn2['do_unpad']:
        Y_2_c = B.unpad(Y_2_c, **DTn2['unpad_kwargs'])
    trim_tm = DTn2['trim_tm']

    return Y_2_c, trim_tm


def _energy_correction(x, factor):
    """See `_compute_energy_correction_factor` in `scat_utils_jtfs.py`."""
    x *= factor
    return x


__all__ = ['timefrequency_scattering1d']
