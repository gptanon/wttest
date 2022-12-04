# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------


def timefrequency_scattering1d(
        x, unpad, backend, J, log2_T, psi1_f, psi2_f, phi_f, scf,
        pad_left=0, pad_right=0, ind_start=None, ind_end=None,
        oversampling=0, oversampling_fr=0, average=True, out_type='array'):
    out_3D = True

    # pack for later
    B = backend
    average_fr = scf.average_fr
    N = x.shape[-1]
    commons = (B, scf, oversampling_fr, average_fr, out_3D, oversampling,
               average, unpad, log2_T, phi_f, ind_start, ind_end, N)

    out_S_1_tm = []
    out_S_2 = {'psi_t * psi_f': [[], []]}

    # pad to a dyadic size and make it complex
    U_0 = B.pad(x, pad_left, pad_right)
    # compute the Fourier transform
    U_0_hat = B.r_fft(U_0)

    # First order ############################################################
    U_1_hat_list = []
    for n1 in range(len(psi1_f)):
        # Convolution + subsampling
        j1 = psi1_f[n1]['j']
        sub1_adj = min(j1, log2_T) if average else j1
        k1 = max(sub1_adj - oversampling, 0)

        U_1_c = B.multiply(U_0_hat, psi1_f[n1][0])
        U_1_hat = B.subsample_fourier(U_1_c, 2**j1)
        U_1_c = B.ifft(U_1_hat)

        # Take the modulus
        U_1_m = B.modulus(U_1_c)

        U_1_hat = B.r_fft(U_1_m)
        U_1_hat_list.append(U_1_hat)

        if average:
            # Lowpass filtering
            k1_log2_T = max(log2_T - k1 - oversampling, 0)
            S_1_c = B.multiply(U_1_hat, phi_f[k1])
            S_1_hat = B.subsample_fourier(S_1_c, 2**k1_log2_T)
            S_1_r = B.ifft_r(S_1_hat)

            # unpad & append to output
            S_1 = unpad(S_1_r, ind_start[k1_log2_T + k1],
                        ind_end[k1_log2_T + k1])
        else:
            S_1 = unpad(U_1_m, ind_start[k1], ind_end[k1])
        out_S_1_tm.append({'coef': S_1, 'j': j1, 'n': n1})

    ##########################################################################
    # Joint scattering: separable convolutions (along time & freq), and low-pass
    # `U1 * (psi_t * psi_f)` (up & down), and `U1 * (psi_t * phi_f)`
    for n2 in range(len(psi2_f)):
        j2 = psi2_f[n2]['j']
        if j2 == 0:
            continue

        Y_2_list = []
        # Wavelet transform over time
        for n1 in range(len(psi1_f)):
            # Retrieve first-order coefficient in the list
            j1 = psi1_f[n1]['j']
            if j1 >= j2:
                continue
            U_1_hat = U_1_hat_list[n1]

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
            Y_2_list.append(Y_2_c)
        assert len(Y_2_list) == scf.N_frs[n2], (len(Y_2_list), scf.N_frs[n2])

        # frequential pad
        pad_fr = scf.J_pad_fr
        Y_2_arr = _right_pad(Y_2_list, pad_fr, scf, B)

        # map to Fourier domain to prepare for conv along freq
        Y_2_hat = B.fft(Y_2_arr, axis=-2)

        # Transform over frequency + low-pass, for both spins ############
        # `* psi_f` part of `U1 * (psi_t * psi_f)`
        _frequency_scattering(Y_2_hat, j2, n2, pad_fr, k1_plus_k2,
                              commons, out_S_2['psi_t * psi_f'])

    ##########################################################################
    # pack outputs & return
    out = {}
    out['S1'] = out_S_1_tm
    out['psi_t * psi_f_up'] = out_S_2['psi_t * psi_f'][0]
    out['psi_t * psi_f_dn'] = out_S_2['psi_t * psi_f'][1]

    # concat
    if out_type == 'array':
        if out_3D:
            # cannot concatenate `S0` & `S1` with joint slices, return separately,
            # even if 'list' for consistency
            o0 = [c for k, v in out.items() for c in v
                  if k in ('S0', 'S1')]
            o1 = [c for k, v in out.items() for c in v
                  if k not in ('S0', 'S1')]

            out_0 = B.concatenate([c['coef'] for c in o0], axis=1)
            out_1 = B.concatenate([c['coef'] for c in o1], axis=1,
                                  keep_cat_dim=True)
            out = (out_0, out_1)
    return out


def _frequency_scattering(Y_2_hat, j2, n2, pad_fr, k1_plus_k2, commons,
                          out_S_2):
    (B, scf, out_exclude, _, _, oversampling_fr, average_fr, out_3D,
     paths_exclude, *_) = commons

    # NOTE: to compute up, we convolve with *down*, since `U1` is time-reversed
    # relative to the sampling of the wavelets.
    psi1_f_frs = (scf.psi1_f_fr_dn, scf.psi1_f_fr_up)
    spins = (1, -1)

    # Transform over frequency + low-pass, for both spins (if `spin_down`)
    for s1_fr, (spin, psi1_f_fr) in enumerate(zip(spins, psi1_f_frs)):
        for n1_fr in range(len(psi1_f_fr)):
            j1_fr = psi1_f_fr[n1_fr]['j']

            # compute subsampling
            total_conv_stride_over_U1 = scf.log2_F

            # Maximum permitted subsampling before conv w/ `phi_f_fr`.
            # This avoids distorting `phi` (aliasing)
            max_subsample_before_phi_fr = scf.log2_F
            sub_adj = min(j1_fr, max_subsample_before_phi_fr);
            n1_fr_subsample = max(sub_adj - oversampling_fr, 0);

            # Wavelet transform over frequency
            Y_fr_c = B.multiply(Y_2_hat, psi1_f_fr[n1_fr][0])
            Y_fr_hat = B.subsample_fourier(Y_fr_c, 2**n1_fr_subsample, axis=-2)
            Y_fr_c = B.ifft(Y_fr_hat, axis=-2)

            # Modulus
            U_2_m = B.modulus(Y_fr_c)

            # Convolve by Phi = phi_t * phi_f, unpad
            S_2 = _joint_lowpass(
                U_2_m, n2, n1_fr, n1_fr_subsample, k1_plus_k2,
                total_conv_stride_over_U1, commons)

            # append to out
            out_S_2[s1_fr].append(
                {'coef': S_2, 'j': (j2, j1_fr), 'n': (n2, n1_fr),
                 'spin': (spin,)})


def _joint_lowpass(U_2_m, n2, n1_fr, n1_fr_subsample, k1_plus_k2,
                   total_conv_stride_over_U1, commons):
    (B, scf, oversampling_fr, average_fr, out_3D, oversampling, average,
     unpad, log2_T, phi_f, ind_start, ind_end, N) = commons
    assert average_fr

    # compute subsampling logic ##############################################
    lowpass_subsample_fr = max(total_conv_stride_over_U1 - n1_fr_subsample
                               - oversampling_fr, 0)

    # freq params
    if out_3D:
        # Unpad length must be same for all `(n2, n1_fr)`; this is determined by
        # the longest N_fr, hence we compute the reference quantities there.
        stride_ref = max(scf.log2_F - oversampling_fr, 0)
        ind_start_fr = scf.ind_start_fr_max[stride_ref]
        ind_end_fr   = scf.ind_end_fr_max[  stride_ref]

    # freq lowpassing ########################################################
    if average_fr:
        # Low-pass filtering over frequency
        phi_fr = scf.phi_f_fr[n1_fr_subsample]
        U_2_hat = B.r_fft(U_2_m, axis=-2)
        S_2_fr_c = B.multiply(U_2_hat, phi_fr)
        S_2_fr_hat = B.subsample_fourier(S_2_fr_c,
                                         2**lowpass_subsample_fr, axis=-2)
        S_2_fr = B.ifft_r(S_2_fr_hat, axis=-2)

    S_2_fr = unpad(S_2_fr, ind_start_fr, ind_end_fr, axis=-2)

    # time lowpassing ########################################################
    if average:
        # Low-pass filtering over time
        k2_log2_T = max(log2_T - k1_plus_k2 - oversampling, 0)
        U_2_hat = B.r_fft(S_2_fr)
        S_2_c = B.multiply(U_2_hat, phi_f[k1_plus_k2])
        S_2_hat = B.subsample_fourier(S_2_c, 2**k2_log2_T)
        S_2_r = B.ifft_r(S_2_hat)
        total_conv_stride_tm = k1_plus_k2 + k2_log2_T
    else:
        total_conv_stride_tm = k1_plus_k2
        S_2_r = S_2_fr
    ind_start_tm = ind_start[total_conv_stride_tm]
    ind_end_tm   = ind_end[total_conv_stride_tm]

    # `not average` and `n2 == -1` already unpadded
    S_2 = unpad(S_2_r, ind_start_tm, ind_end_tm)

    return S_2


#### helper methods ##########################################################
def _right_pad(coeff_list, pad_fr, scf, B):
    # zero-pad
    zero_row = B.zeros_like(coeff_list[0])
    zero_rows = [zero_row] * (2**pad_fr - len(coeff_list))
    return B.concatenate(coeff_list + zero_rows, axis=1)


__all__ = ['timefrequency_scattering1d']
