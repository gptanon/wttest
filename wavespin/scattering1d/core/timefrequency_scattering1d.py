# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import math
from ..backend.agnostic_backend import unpad_dyadic


def timefrequency_scattering1d(
        x, unpad, backend, J, log2_T, psi1_f, psi2_f, phi_f, scf, pad_fn,
        pad_mode='zero', pad_left=0, pad_right=0, ind_start=None, ind_end=None,
        oversampling=0, oversampling_fr=0, aligned=True, F_kind='average',
        average=True, average_global=None, average_global_phi=None,
        out_type='array', out_3D=False, out_exclude=None, paths_exclude=None,
        api_pair_order=None):
    """
    Main function implementing the Joint Time-Frequency Scattering transform.

    For a minimalistic, easier to understand implementation, see
    `examples/jtfs-min/`.

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
    average_fr = scf.average_fr
    if out_exclude is None:
        out_exclude = []
    N = x.shape[-1]
    commons = (B, scf, out_exclude, aligned, F_kind, oversampling_fr, average_fr,
               out_3D, paths_exclude, oversampling, average, average_global,
               average_global_phi, unpad, log2_T, phi_f, ind_start, ind_end, N)

    out_S_0 = []
    out_S_1_tm = []
    out_S_1 = {'phi_t * phi_f': []}
    out_S_2 = {'psi_t * psi_f': [[], []],
               'psi_t * phi_f': [],
               'phi_t * psi_f': [[]]}

    # pad to a dyadic size and make it complex
    U_0 = pad_fn(x)
    # compute the Fourier transform
    U_0_hat = B.r_fft(U_0)

    # for later
    J_pad = math.log2(U_0.shape[-1])
    commons2 = (average, log2_T, J, J_pad, N, ind_start, ind_end, unpad, phi_f,
                pad_mode)

    # Zeroth order ###########################################################
    if 'S0' not in out_exclude:
        if average_global:
            k0 = log2_T
            S_0 = B.mean(U_0, axis=-1)
        elif average:
            k0 = max(log2_T - oversampling, 0)
            S_0_c = B.multiply(U_0_hat, phi_f[0][0])
            S_0_hat = B.subsample_fourier(S_0_c, 2**k0)
            S_0_r = B.ifft_r(S_0_hat)
            S_0 = unpad(S_0_r, ind_start[0][k0], ind_end[0][k0])
        else:
            S_0 = x
        if average:
            S_0 *= B.sqrt(2**k0, dtype=S_0.dtype)  # subsampling energy correction
        out_S_0.append({'coef': S_0,
                        'j': (log2_T,) if average else (),
                        'n': (-1,)     if average else (),
                        'spin': (),
                        'stride': (k0,) if average else (), })

    # First order ############################################################
    def compute_U_1(n1, k1):
        U_1_c = B.multiply(U_0_hat, psi1_f[n1][0])
        U_1_hat = B.subsample_fourier(U_1_c, 2**k1)
        U_1_c = B.ifft(U_1_hat)

        # Modulus
        U_1_m = B.modulus(U_1_c)

        # Map to Fourier domain
        U_1_hat = B.r_fft(U_1_m)
        return U_1_hat, U_1_m

    include_phi_t = any(pair not in out_exclude for pair in
                        ('phi_t * phi_f', 'phi_t * psi_f'))

    U_1_hat_list, S_1_tm_list = [], []
    for n1 in range(len(psi1_f)):
        # Convolution + subsampling
        j1 = psi1_f[n1]['j']
        sub1_adj = min(j1, log2_T) if average else j1
        k1 = max(sub1_adj - oversampling, 0)

        U_1_hat, U_1_m = compute_U_1(n1, k1)
        U_1_hat_list.append(U_1_hat)

        # if `k1` is used from this point, treat as if `average=True`
        sub1_adj_avg = min(j1, log2_T)
        k1_avg = max(sub1_adj_avg - oversampling, 0)
        if average or include_phi_t:
            k1_log2_T = (max(log2_T - k1_avg - oversampling, 0)
                         if not average_global_phi else log2_T - k1_avg)
            ind_start_tm_avg = ind_start[0][k1_log2_T + k1_avg]
            ind_end_tm_avg   = ind_end[  0][k1_log2_T + k1_avg]
            if not average_global_phi:
                if k1 != k1_avg:
                    # must recompute U_1_hat
                    U_1_hat_avg, _ = compute_U_1(n1, k1_avg)
                else:
                    U_1_hat_avg = U_1_hat
                # Low-pass filtering over time
                S_1_c = B.multiply(U_1_hat_avg, phi_f[0][k1_avg])

                S_1_hat = B.subsample_fourier(S_1_c, 2**k1_log2_T)
                S_1_avg = B.ifft_r(S_1_hat)
                # unpad since we're fully done with convolving over time
                S_1_avg = unpad(S_1_avg, ind_start_tm_avg, ind_end_tm_avg)
                total_conv_stride_tm_avg = k1_avg + k1_log2_T
            else:
                # Average directly
                S_1_avg = B.mean(U_1_m, axis=-1)
                total_conv_stride_tm_avg = log2_T

        if 'S1' not in out_exclude:
            if average:
                ind_start_tm, ind_end_tm = ind_start_tm_avg, ind_end_tm_avg
            if average_global:
                S_1_tm = S_1_avg
            elif average:
                # Unpad averaged
                S_1_tm = S_1_avg
            else:
                # Unpad unaveraged
                ind_start_tm, ind_end_tm = ind_start[0][k1], ind_end[0][k1]
                S_1_tm = unpad(U_1_m, ind_start_tm, ind_end_tm)
            total_conv_stride_tm = (total_conv_stride_tm_avg if average else
                                    k1)

            # energy correction due to stride & inexact unpad length
            S_1_tm = _energy_correction(S_1_tm, B,
                                        param_tm=(N, ind_start_tm,
                                                  ind_end_tm,
                                                  total_conv_stride_tm))
            out_S_1_tm.append({'coef': S_1_tm,
                               'j': (j1,), 'n': (n1,), 'spin': (),
                               'stride': (total_conv_stride_tm,)})

            # since tensorflow won't update it in `_energy_correction`
            if average:
                S_1_avg = S_1_tm

        # append for further processing
        if include_phi_t:
            # energy correction, if not done
            S_1_avg_energy_corrected = bool(average and
                                            'S1' not in out_exclude)
            if not S_1_avg_energy_corrected:
                S_1_avg = _energy_correction(
                    S_1_avg, B, param_tm=(N, ind_start_tm_avg,
                                          ind_end_tm_avg,
                                          total_conv_stride_tm_avg))
            S_1_tm_list.append(S_1_avg)

    # Frequential averaging over time averaged coefficients ##################
    # `U1 * (phi_t * phi_f)` pair
    if include_phi_t:
        # zero-pad along frequency
        pad_fr = scf.J_pad_frs_max
        if len(S_1_tm_list) < 2**pad_fr:
            # usual case
            S_1_tm = _right_pad(S_1_tm_list, pad_fr, scf, B)
        else:
            # handle rare edge case (see `_J_pad_fr_fo` docs)
            assert (scf.max_pad_factor_fr is not None and
                    scf.max_pad_factor_fr[0] == 0), scf.max_pad_factor_fr
            S_1_tm = S_1_tm_list[:2**pad_fr]

        if (('phi_t * phi_f' not in out_exclude and
             not scf.average_fr_global_phi) or
                'phi_t * psi_f' not in out_exclude):
            # map frequency axis to Fourier domain
            S_1_tm_hat = B.r_fft(S_1_tm, axis=-2)

    if 'phi_t * phi_f' not in out_exclude:
        lowpass_subsample_fr = 0  # no lowpass after lowpass

        if scf.average_fr_global_phi:
            # take mean along frequency directly
            S_1 = B.mean(S_1_tm, axis=-2)
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

            # Low-pass filtering over frequency
            phi_fr = scf.phi_f_fr[log2_F_phi_diff][pad_diff][0]
            S_1_c = B.multiply(S_1_tm_hat, phi_fr)
            S_1_hat = B.subsample_fourier(S_1_c, 2**n1_fr_subsample, axis=-2)
            S_1_r = B.ifft_r(S_1_hat, axis=-2)

        # compute for unpad & energy correction
        _stride = n1_fr_subsample + lowpass_subsample_fr
        if out_3D:
            ind_start_fr = scf.ind_start_fr_max[_stride]
            ind_end_fr   = scf.ind_end_fr_max[  _stride]
        else:
            ind_start_fr = scf.ind_start_fr[-1][_stride]
            ind_end_fr   = scf.ind_end_fr[-1][  _stride]

        # Unpad frequency
        if not scf.average_fr_global_phi:
            S_1 = unpad(S_1_r, ind_start_fr, ind_end_fr, axis=-2)

        # set reference for later
        total_conv_stride_over_U1_realized = (n1_fr_subsample +
                                              lowpass_subsample_fr)

        # energy correction due to stride & inexact unpad indices
        # time already done
        S_1 = _energy_correction(S_1, B,
                                 param_fr=(scf.N_frs_max,
                                           ind_start_fr, ind_end_fr,
                                           total_conv_stride_over_U1_realized))

        scf.__total_conv_stride_over_U1 = total_conv_stride_over_U1_realized
        # append to out with meta
        stride = (total_conv_stride_over_U1_realized, log2_T)
        out_S_1['phi_t * phi_f'].append({
            'coef': S_1, 'j': (log2_T, log2_F_phi), 'n': (-1, -1),
            'spin': (0,), 'stride': stride})
    else:
        scf.__total_conv_stride_over_U1 = -1

    ##########################################################################
    # Joint scattering: separable convolutions (along time & freq), and low-pass
    # `U1 * (psi_t * psi_f)` (up & down), and `U1 * (psi_t * phi_f)`
    skip_spinned = bool('psi_t * psi_f_up' in out_exclude and
                        'psi_t * psi_f_dn' in out_exclude)
    if not (skip_spinned and 'psi_t * phi_f' in out_exclude):
        for n2 in range(len(psi2_f)):
            if (n2 not in scf.paths_include_build or
                n2 in paths_exclude['n2']):
                continue
            j2 = psi2_f[n2]['j']

            Y_2_list = []
            # Wavelet transform over time
            for n1 in range(len(psi1_f)):
                # Retrieve first-order coefficient in the list
                if n1 not in scf.paths_include_build[n2]:
                    continue
                U_1_hat = U_1_hat_list[n1]
                j1 = psi1_f[n1]['j']

                # what we subsampled in 1st-order
                sub1_adj = min(j1, log2_T) if average else j1
                k1 = max(sub1_adj - oversampling, 0)
                # what we subsample now in 2nd
                sub2_adj = min(j2, log2_T) if average else j2
                k2 = max(sub2_adj - k1 - oversampling, 0)

                # Convolution and downsampling
                Y_2_c = B.multiply(U_1_hat, psi2_f[n2][k1])
                Y_2_hat = B.subsample_fourier(Y_2_c, 2**k2)
                Y_2_c = B.ifft(Y_2_hat)

                # sum is same for all `n1`
                k1_plus_k2 = k1 + k2
                Y_2_c, trim_tm = _maybe_unpad_time(Y_2_c, k1_plus_k2, commons2)
                Y_2_list.append(Y_2_c)
            assert len(Y_2_list) == scf.N_frs[n2], (len(Y_2_list), scf.N_frs[n2])

            # frequential pad
            scale_diff = scf.scale_diffs[n2]
            pad_fr = scf.J_pad_frs[scale_diff]

            if scf.pad_mode_fr == 'custom':
                Y_2_arr = scf.pad_fn_fr(Y_2_list, pad_fr, scf, B)
            else:
                Y_2_arr = _right_pad(Y_2_list, pad_fr, scf, B)

            # temporal pad modification
            if pad_mode == 'reflect' and average:
                # `=` since tensorflow makes copy
                Y_2_arr = B.conj_reflections(Y_2_arr,
                                             ind_start[trim_tm][k1_plus_k2],
                                             ind_end[  trim_tm][k1_plus_k2],
                                             k1_plus_k2, N,
                                             pad_left, pad_right, trim_tm)

            # swap axes & map to Fourier domain to prepare for conv along freq
            Y_2_hat = B.fft(Y_2_arr, axis=-2)

            # Transform over frequency + low-pass, for both spins ############
            # `* psi_f` part of `U1 * (psi_t * psi_f)`
            if not skip_spinned:
                _frequency_scattering(Y_2_hat, j2, n2, pad_fr, k1_plus_k2,
                                      trim_tm, commons, out_S_2['psi_t * psi_f'])

            # Low-pass over frequency ########################################
            # `* phi_f` part of `U1 * (psi_t * phi_f)`
            if 'psi_t * phi_f' not in out_exclude:
                _frequency_lowpass(Y_2_hat, Y_2_arr, j2, n2, pad_fr, k1_plus_k2,
                                   trim_tm, commons, out_S_2['psi_t * phi_f'])

    ##########################################################################
    # `U1 * (phi_t * psi_f)`
    if 'phi_t * psi_f' not in out_exclude:
        # take largest subsampling factor
        j2 = log2_T
        k1_plus_k2 = (max(log2_T - oversampling, 0) if not average_global_phi else
                      log2_T)
        pad_fr = scf.J_pad_frs_max
        # n2_time = U_0.shape[-1] // 2**max(j2 - oversampling, 0)

        # reuse from first-order scattering
        Y_2_hat = S_1_tm_hat

        # Transform over frequency + low-pass
        # `* psi_f` part of `U1 * (phi_t * psi_f)`
        _frequency_scattering(Y_2_hat, j2, -1, pad_fr, k1_plus_k2, 0, commons,
                              out_S_2['phi_t * psi_f'], spin_down=False)

    ##########################################################################
    # pack outputs & return
    out = {}
    out['S0'] = out_S_0
    out['S1'] = out_S_1_tm
    out['phi_t * phi_f'] = out_S_1['phi_t * phi_f']
    out['phi_t * psi_f'] = out_S_2['phi_t * psi_f'][0]
    out['psi_t * phi_f'] = out_S_2['psi_t * phi_f']
    out['psi_t * psi_f_up'] = out_S_2['psi_t * psi_f'][0]
    out['psi_t * psi_f_dn'] = out_S_2['psi_t * psi_f'][1]
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
                if out_3D:
                    # stack joint slices, preserve 3D structure
                    out[k] = B.concatenate([c['coef'] for c in v], axis=1,
                                           keep_cat_dim=True)
                else:
                    # flatten joint slices, return 2D
                    out[k] = B.concatenate([c['coef'] for c in v], axis=1)

    elif out_type == 'dict:list':
        pass  # already done

    else:
        if out_3D:
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
            if out_3D:
                out_0 = B.concatenate([c['coef'] for c in o0], axis=1)
                out_1 = B.concatenate([c['coef'] for c in o1], axis=1,
                                      keep_cat_dim=True)
                out = (out_0, out_1)
            else:
                out = B.concatenate([c['coef'] for c in o], axis=1)
        elif out_type == 'list':
            if out_3D:
                out = (o0, o1)
            else:
                out = o

    return out


def _frequency_scattering(Y_2_hat, j2, n2, pad_fr, k1_plus_k2, trim_tm, commons,
                          out_S_2, spin_down=True):
    (B, scf, out_exclude, _, _, oversampling_fr, average_fr, out_3D,
     paths_exclude, *_) = commons

    # NOTE: to compute up, we convolve with *down*, since `U1` is time-reversed
    # relative to the sampling of the wavelets.
    psi1_f_frs, spins = [], []
    if ('psi_t * psi_f_up' not in out_exclude or
            'psi_t * phi_f' not in out_exclude):
        psi1_f_frs.append(scf.psi1_f_fr_dn)
        spins.append(1 if spin_down else 0)
    if spin_down and 'psi_t * psi_f_dn' not in out_exclude:
        psi1_f_frs.append(scf.psi1_f_fr_up)
        spins.append(-1)

    scale_diff = scf.scale_diffs[n2]
    psi_id = scf.psi_ids[scale_diff]
    pad_diff = scf.J_pad_frs_max_init - pad_fr

    n1_fr_subsamples = scf.n1_fr_subsamples['spinned'][scale_diff]
    log2_F_phi_diffs = scf.log2_F_phi_diffs['spinned'][scale_diff]

    # Transform over frequency + low-pass, for both spins (if `spin_down`)
    for s1_fr, (spin, psi1_f_fr) in enumerate(zip(spins, psi1_f_frs)):
        for n1_fr in range(len(psi1_f_fr[psi_id])):
            if n1_fr in paths_exclude.get('n1_fr', {}):
                continue
            j1_fr = psi1_f_fr['j'][psi_id][n1_fr]

            # compute subsampling
            total_conv_stride_over_U1 = scf.total_conv_stride_over_U1s[
                scale_diff][n1_fr]
            log2_F_phi_diff = log2_F_phi_diffs[n1_fr]
            n1_fr_subsample = max(n1_fr_subsamples[n1_fr] - oversampling_fr, 0)

            # Wavelet transform over frequency
            Y_fr_c = B.multiply(Y_2_hat, psi1_f_fr[psi_id][n1_fr])
            Y_fr_hat = B.subsample_fourier(Y_fr_c, 2**n1_fr_subsample, axis=-2)
            Y_fr_c = B.ifft(Y_fr_hat, axis=-2)

            # Modulus
            U_2_m = B.modulus(Y_fr_c)

            # Convolve by Phi = phi_t * phi_f, unpad
            S_2, stride = _joint_lowpass(
                U_2_m, n2, n1_fr, pad_diff, n1_fr_subsample, log2_F_phi_diff,
                k1_plus_k2, total_conv_stride_over_U1, trim_tm, commons)

            # append to out
            out_S_2[s1_fr].append(
                {'coef': S_2, 'j': (j2, j1_fr), 'n': (n2, n1_fr),
                 'spin': (spin,), 'stride': stride})


def _frequency_lowpass(Y_2_hat, Y_2_arr, j2, n2, pad_fr, k1_plus_k2, trim_tm,
                       commons, out_S_2):
    B, scf, _, _, _, oversampling_fr, average_fr, *_ = commons

    pad_diff = scf.J_pad_frs_max_init - pad_fr

    if scf.average_fr_global_phi:
        Y_fr_c = B.mean(Y_2_arr, axis=-2)
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
                              oversampling_fr, 0)
        # scale params
        log2_F_phi = scf.log2_F_phis['phi'][scale_diff]
        log2_F_phi_diff = scf.log2_F_phi_diffs['phi'][scale_diff]

        # convolve
        Y_fr_c = B.multiply(Y_2_hat, scf.phi_f_fr[log2_F_phi_diff][pad_diff][0])
        Y_fr_hat = B.subsample_fourier(Y_fr_c, 2**n1_fr_subsample, axis=-2)
        Y_fr_c = B.ifft(Y_fr_hat, axis=-2)

    # Modulus
    U_2_m = B.modulus(Y_fr_c)

    # Convolve by Phi = phi_t * phi_f
    S_2, stride = _joint_lowpass(U_2_m, n2, -1, pad_diff,
                                 n1_fr_subsample, log2_F_phi_diff, k1_plus_k2,
                                 total_conv_stride_over_U1_phi, trim_tm, commons)

    out_S_2.append({'coef': S_2, 'j': (j2, log2_F_phi), 'n': (n2, -1),
                    'spin': (0,), 'stride': stride})


def _joint_lowpass(U_2_m, n2, n1_fr, pad_diff, n1_fr_subsample, log2_F_phi_diff,
                   k1_plus_k2, total_conv_stride_over_U1, trim_tm, commons):
    (B, scf, _, aligned, F_kind, oversampling_fr, average_fr, out_3D, _,
     oversampling, average, average_global, average_global_phi, unpad, log2_T,
     phi_f, ind_start, ind_end, N) = commons

    # compute subsampling logic ##############################################
    global_averaged_fr = (scf.average_fr_global if n1_fr != -1 else
                          scf.average_fr_global_phi)
    if global_averaged_fr:
        lowpass_subsample_fr = total_conv_stride_over_U1 - n1_fr_subsample
    elif average_fr:
        lowpass_subsample_fr = max(total_conv_stride_over_U1 - n1_fr_subsample
                                   - oversampling_fr, 0)
    else:
        lowpass_subsample_fr = 0

    # compute freq unpad params ##############################################
    do_averaging    = bool(average    and n2    != -1)
    do_averaging_fr = bool(average_fr and n1_fr != -1)
    total_conv_stride_over_U1_realized = n1_fr_subsample + lowpass_subsample_fr

    # freq params
    if out_3D:
        # Unpad length must be same for all `(n2, n1_fr)`; this is determined by
        # the longest N_fr, hence we compute the reference quantities there.
        stride_ref = scf.total_conv_stride_over_U1s[0][0]
        stride_ref = max(stride_ref - oversampling_fr, 0)
        ind_start_fr = scf.ind_start_fr_max[stride_ref]
        ind_end_fr   = scf.ind_end_fr_max[  stride_ref]
    else:
        _stride = total_conv_stride_over_U1_realized
        ind_start_fr = scf.ind_start_fr[n2][_stride]
        ind_end_fr   = scf.ind_end_fr[  n2][_stride]

    # unpad early if possible ################################################
    # `not average` and `n2 == -1` already unpadded
    if not do_averaging:
        pass
    if not do_averaging_fr:
        U_2_m = unpad(U_2_m, ind_start_fr, ind_end_fr, axis=-2)
    unpadded_tm, unpadded_fr = (not do_averaging), (not do_averaging_fr)

    # freq lowpassing ########################################################
    if do_averaging_fr:
        if scf.average_fr_global:
            S_2_fr = B.mean(U_2_m, axis=-2)
        elif average_fr:
            if F_kind == 'average':
                # Low-pass filtering over frequency
                phi_fr = scf.phi_f_fr[log2_F_phi_diff][pad_diff][n1_fr_subsample]
                U_2_hat = B.r_fft(U_2_m, axis=-2)
                S_2_fr_c = B.multiply(U_2_hat, phi_fr)
                S_2_fr_hat = B.subsample_fourier(S_2_fr_c,
                                                 2**lowpass_subsample_fr, axis=-2)
                S_2_fr = B.ifft_r(S_2_fr_hat, axis=-2)
            elif F_kind == 'decimate':
                assert oversampling_fr == 0  # future todo
                if lowpass_subsample_fr != 0:
                    S_2_fr = scf.decimate(U_2_m, 2**lowpass_subsample_fr, axis=-2)
                else:
                    S_2_fr = U_2_m
    else:
        S_2_fr = U_2_m

    # unpad only if input isn't global averaged
    do_unpad_fr = bool(not global_averaged_fr and not unpadded_fr)
    if do_unpad_fr:
        S_2_fr = unpad(S_2_fr, ind_start_fr, ind_end_fr, axis=-2)

    # time lowpassing ########################################################
    if do_averaging:  # do 1st, future todo
        if average_global:
            S_2_r = B.mean(S_2_fr, axis=-1)
            total_conv_stride_tm = log2_T
        elif average:
            # Low-pass filtering over time
            k2_log2_T = max(log2_T - k1_plus_k2 - oversampling, 0)
            U_2_hat = B.r_fft(S_2_fr)
            S_2_c = B.multiply(U_2_hat, phi_f[trim_tm][k1_plus_k2])
            S_2_hat = B.subsample_fourier(S_2_c, 2**k2_log2_T)
            S_2_r = B.ifft_r(S_2_hat)
            total_conv_stride_tm = k1_plus_k2 + k2_log2_T
    else:
        total_conv_stride_tm = k1_plus_k2
        S_2_r = S_2_fr
    ind_start_tm = ind_start[trim_tm][total_conv_stride_tm]
    ind_end_tm   = ind_end[  trim_tm][total_conv_stride_tm]

    # `not average` and `n2 == -1` already unpadded
    if not average_global and not unpadded_tm:
        S_2_r = unpad(S_2_r, ind_start_tm, ind_end_tm)
    S_2 = S_2_r

    # energy correction ######################################################
    param_tm = (N, ind_start_tm, ind_end_tm, total_conv_stride_tm)
    param_fr = (scf.N_frs[n2], ind_start_fr,
                ind_end_fr, total_conv_stride_over_U1_realized)
    # correction due to stride & inexact unpad indices
    S_2 = (_energy_correction(S_2, B, param_tm, param_fr) if n2 != -1 else
           # `n2=-1` already did time
           _energy_correction(S_2, B, param_fr=param_fr, phi_t_psi_f=True))

    # sanity checks (see "Subsampling, padding") #############################
    if aligned:
        if not global_averaged_fr:
            # `total_conv_stride_over_U1` renamed; comment for searchability
            if scf.__total_conv_stride_over_U1 == -1:
                scf.__total_conv_stride_over_U1 = (  # set if not set yet
                    total_conv_stride_over_U1_realized)
            if not scf.average_fr:
                assert total_conv_stride_over_U1_realized == 0
            else:
                a = total_conv_stride_over_U1_realized
                b = scf.__total_conv_stride_over_U1
                assert a == b, (a, b)

    # ensure we didn't lose info (over-trim)
    unpad_len_fr = S_2.shape[-2]
    assert (unpad_len_fr >= scf.N_frs[n2] / 2**total_conv_stride_over_U1_realized
            ), (unpad_len_fr, scf.N_frs, total_conv_stride_over_U1_realized)
    # ensure we match specified unpad len (possible to fail by under-padding)
    if do_unpad_fr:
        unpad_len_fr_expected = ind_end_fr - ind_start_fr
        assert unpad_len_fr_expected == unpad_len_fr, (
            unpad_len_fr_expected, unpad_len_fr)

    stride = (total_conv_stride_over_U1_realized, total_conv_stride_tm)
    return S_2, stride


#### helper methods ##########################################################
def _right_pad(coeff_list, pad_fr, scf, B):
    if scf.pad_mode_fr == 'conj-reflect-zero':
        return _pad_conj_reflect_zero(coeff_list, pad_fr, scf.N_frs_max_all, B)
    # zero-pad
    zero_row = B.zeros_like(coeff_list[0])
    zero_rows = [zero_row] * (2**pad_fr - len(coeff_list))
    return B.concatenate(coeff_list + zero_rows, axis=1)


def _pad_conj_reflect_zero(coeff_list, pad_fr, N_frs_max_all, B):
    n_coeffs_input = len(coeff_list)  # == N_fr
    zero_row = B.zeros_like(coeff_list[0])
    padded_len = 2**pad_fr
    # first zero pad, then reflect remainder (including zeros as appropriate)
    n_zeros = min(N_frs_max_all - n_coeffs_input,  # never need more than this
                  padded_len - n_coeffs_input)     # cannot exceed `padded_len`
    zero_rows = [zero_row] * n_zeros

    coeff_list_new = coeff_list + zero_rows
    right_pad = max((padded_len - n_coeffs_input) // 2, n_zeros)
    left_pad  = padded_len - right_pad - n_coeffs_input

    # right pad
    right_rows = zero_rows
    idx = -2
    reflect = False
    while len(right_rows) < right_pad:
        c = coeff_list_new[idx]
        c = c if reflect else B.conj(c)
        right_rows.append(c)
        if idx in (-1, -len(coeff_list_new)):
            reflect = not reflect
        idx += 1 if reflect else -1

    # (circ-)left pad
    left_rows = []
    idx = - (len(coeff_list_new) - 1)
    reflect = False
    while len(left_rows) < left_pad:
        c = coeff_list_new[idx]
        c = c if reflect else B.conj(c)
        left_rows.append(c)
        if idx in (-1, -len(coeff_list_new)):
            reflect = not reflect
        idx += -1 if reflect else 1
    left_rows = left_rows[::-1]

    return B.concatenate(coeff_list + right_rows + left_rows, axis=1)


def _maybe_unpad_time(Y_2_c, k1_plus_k2, commons2):
    from ..filter_bank import N_and_pad_2_J_pad
    (average, log2_T, J, J_pad, N, ind_start, ind_end, unpad, phi_f, pad_mode
     ) = commons2

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
            Y_2_c = unpad_dyadic(Y_2_c, end - start, 2**J_pad, 2**pad_log2_T,
                                 k1_plus_k2)
        trim_tm = amount_can_unpad_now
    elif not average:
        # can unpad fully, no further processing along time
        Y_2_c = unpad(Y_2_c, start, end)
        # no longer applicable, but still needed for computing final time length
        trim_tm = 0
    else:
        # no unpadding, use full length of `phi_f`
        trim_tm = 0

    return Y_2_c, trim_tm


def _energy_correction(x, B, param_tm=None, param_fr=None, phi_t_psi_f=False):
    """Enables `||jtfs(x)|| = ||x||`.

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
    """
    energy_correction_tm, energy_correction_fr = 1, 1
    if param_tm is not None:
        N, ind_start_tm, ind_end_tm, total_conv_stride_tm = param_tm
        # time energy correction due to integer-rounded unpad indices
        unpad_len_exact = N / 2**total_conv_stride_tm
        unpad_len = ind_end_tm - ind_start_tm
        if not (unpad_len_exact.is_integer() and unpad_len == unpad_len_exact):
            energy_correction_tm *= B.sqrt(unpad_len_exact / unpad_len,
                                           dtype=x.dtype)
        # compensate for subsampling
        energy_correction_tm *= B.sqrt(2**total_conv_stride_tm, dtype=x.dtype)

    if param_fr is not None:
        (N_fr, ind_start_fr, ind_end_fr, total_conv_stride_over_U1_realized
         ) = param_fr
        # freq energy correction due to integer-rounded unpad indices
        unpad_len_exact = N_fr / 2**total_conv_stride_over_U1_realized
        unpad_len = ind_end_fr - ind_start_fr
        if not (unpad_len_exact.is_integer() and unpad_len == unpad_len_exact):
            energy_correction_fr *= B.sqrt(unpad_len_exact / unpad_len,
                                           dtype=x.dtype)
        # compensate for subsampling
        energy_correction_fr *= B.sqrt(2**total_conv_stride_over_U1_realized,
                                       dtype=x.dtype)

        if phi_t_psi_f:
            # since we only did one spin
            energy_correction_fr *= B.sqrt(2, dtype=x.dtype)

    if energy_correction_tm != 1 or energy_correction_fr != 1:
        x *= (energy_correction_tm * energy_correction_fr)
    return x


__all__ = ['timefrequency_scattering1d']
