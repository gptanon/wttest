# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------


def scattering1d(x, pad_fn, backend, log2_T, psi1_f, psi2_f, phi_f,
                 compute_graph, ind_start, ind_end,
                 oversampling, max_order, average, out_type,
                 average_global, vectorized, vectorized_early_U_1,
                 psi1_f_stacked, jtfs_cfg=None):
    """
    Main function implementing the 1-D scattering transform.
    See `scattering` in `help(wavespin.Scattering1D())`.

    This code is meant to be read alongside `build_compute_graph_tm` in
    `wavespin/scattering1d/scat_utils.py`.

    For an easier to understand equivalent implementation (that's slower), see
    `examples/jtfs-min/jtfs_min/scattering1d/core/scattering1d.py`.
    """
    # pack/unpack stuff ######################################################
    B = backend
    out_S_0, out_S_1, out_S_2 = [], [], []

    (U_1_dict, U_12_dict, keys1_grouped, offsets, n1s_of_n2,
     n_n1s_for_n2_and_k1) = [
         compute_graph[name] for name in
         ('U_1_dict', 'U_12_dict', 'keys1_grouped', 'offsets', 'n1s_of_n2',
          'n_n1s_for_n2_and_k1')]

    # assignment ops may be slower with `False`
    can_assign_directly = bool(B.name not in ('tensorflow', 'jax'))
    commons = (B, log2_T, vectorized, oversampling, ind_start, ind_end, phi_f)

    # handle JTFS if relevant, and other bools -------------------------------
    if jtfs_cfg is None:
        not_jtfs = True

        do_S_0 = True
        do_S_1_avg = bool(average)
        do_S_1_tm = True
        do_S_2 = bool(max_order == 2)
        jtfs_needs_U_2_c = False
        do_U_1_hat = bool((average and not average_global) or max_order > 1)
    else:
        not_jtfs = False
        assert max_order == 2

        do_S_0 = bool(jtfs_cfg['do_S_0'])
        do_S_1_avg = bool(jtfs_cfg['do_S_1_avg'])
        do_S_1_tm = bool(jtfs_cfg['do_S_1_tm'])
        do_S_2 = False
        jtfs_needs_U_2_c = bool(jtfs_cfg['jtfs_needs_U_2_c'])
        do_U_1_hat = jtfs_needs_U_2_c

        if do_S_1_avg:
            S_1_avgs = []
        if jtfs_needs_U_2_c:
            U_2_cs = {}
        average_global_phi = bool(jtfs_cfg['average_global_phi'])

        # since jtfs modifies these per `trim_tm`
        _phi_f = {key: phi_f[key] for key in phi_f if not isinstance(key, int)}
        for key in range(len(phi_f[0])):
            _phi_f[key] = phi_f[0][key]
        phi_f = _phi_f
        ind_start = ind_start[0]
        ind_end = ind_end[0]

    # handle input ###########################################################
    # pad to a dyadic size and make it complex
    U_0 = pad_fn(x)
    # compute the Fourier transform
    U_0_hat = B.r_fft(U_0)

    # Zeroth order ###########################################################
    if do_S_0:
        if average_global:
            k0 = log2_T
        elif average:
            k0 = max(log2_T - oversampling, 0)

        if average_global:
            S_0 = B.mean(U_0, axis=-1)
        elif average:
            S_0_c = B.multiply(U_0_hat, phi_f[0])
            S_0_hat = B.subsample_fourier(S_0_c, 2**k0)
            S_0_r = B.ifft_r(S_0_hat)

            S_0 = B.unpad(S_0_r, ind_start[k0], ind_end[k0])
        else:
            S_0 = x
        out_S_0.append({'coef': S_0,
                        'j': (),
                        'n': ()})

    # First order ############################################################
    # make compute blocks ####################################################
    # preallocate
    U_1_hats_grouped = {}
    for k1 in U_1_dict:
        shape = list(U_0_hat.shape)
        shape[1] = U_1_dict[k1]
        shape[-1] //= 2**k1
        U_1_hats_grouped[k1] = B.zeros_like(U_0_hat, shape)

    # execute compute blocks #################################################
    # U_1_c, maybe U_1_hat ---------------------------------------------------
    if vectorized_early_U_1:
        # multiply `fft(x)` by all filters at once
        U_1_cs_grouped = B.multiply(U_0_hat, psi1_f_stacked)
    else:
        for n1, p1f in enumerate(psi1_f):
            # Convolution + downsampling
            j1 = p1f['j']
            k1 = max(min(j1, log2_T) - oversampling, 0)
            # here it's possible to instead do
            #     sub1_adj = min(j1, log2_T) if average else j1
            #     k1 = max(sub1_adj - oversampling, 0)
            # but this complicates implementation for an uncommon use case

            U_1_c = B.multiply(U_0_hat, p1f[0])
            U_1_hat = B.subsample_fourier(U_1_c, 2**k1)

            # Store coefficient in proper grouping
            offset = offsets[n1]
            if can_assign_directly:
                U_1_hats_grouped[k1][:, n1 - offset] = U_1_hat
            else:
                U_1_hats_grouped[k1] = B.assign_slice(
                    U_1_hats_grouped[k1], U_1_hat, n1 - offset, axis=1)

    # lowpass filtering ######################################################
    # U_1_m, U_1_hat, S_1 ----------------------------------------------------
    for k1 in U_1_dict:
        # U_1_m, U_1_hat -----------------------------------------------------
        # group computation by lengths, controlled by `k1`
        if vectorized:
            # slice coeffs with same subsampled length, controlled by `k1`
            if vectorized_early_U_1:
                start, end = keys1_grouped[k1][0], keys1_grouped[k1][-1] + 1
                U_1_arr = U_1_cs_grouped[:, start:end]
                out = _compute_U_1(U_1_arr, True, do_U_1_hat, B, k1=k1)
            else:
                U_1_arr = U_1_hats_grouped[k1]
                out = _compute_U_1(U_1_arr, False, do_U_1_hat, B)

            if do_U_1_hat:
                U_1_m, U_1_hat = out
                U_1_hats_grouped[k1] = U_1_hat
            else:
                U_1_m = out

        else:
            U_1_arr = U_1_hats_grouped[k1]
            U_1_m = []
            for i in range(U_1_arr.shape[1]):
                out = _compute_U_1(U_1_arr[:, i:i+1], vectorized_early_U_1,
                                   do_U_1_hat, B, k1=k1)
                if do_U_1_hat:
                    U_1_m.append(out[0])
                    if can_assign_directly:
                        U_1_hats_grouped[k1][:, i:i+1] = out[1]
                    else:
                        U_1_hats_grouped[k1] = B.assign_slice(
                            U_1_hats_grouped[k1], out[1], i, axis=1)
                else:
                    U_1_m.append(out)

        # S_1 -----------------------------------------------------------------
        # Lowpass filtering
        completely_skip_S_1 = bool(not not_jtfs and
                                   not (do_S_1_tm or do_S_1_avg))
        if not completely_skip_S_1:
            lowpass_data = dict(commons=commons, k1=k1, U_1_m=U_1_m,
                                U_1_hats_grouped=U_1_hats_grouped)
            if not_jtfs:
                S_1 = _lowpass_first_order(**lowpass_data, average=average,
                                           average_global=average_global)
            else:
                if do_S_1_tm:
                    S_1_tm = _lowpass_first_order(**lowpass_data, average=average,
                                                  average_global=average_global)
                    S_1 = S_1_tm

                # need averaged coefficients...
                if do_S_1_avg:
                    # ... and already have them
                    if average and do_S_1_tm:
                        S_1_avg = S_1_tm
                    # ... but don't have them
                    else:
                        S_1_avg = _lowpass_first_order(
                            **lowpass_data, average=True,
                            average_global=average_global_phi)
                    # append into outputs
                    if vectorized:
                        S_1_avgs.append(S_1_avg)
                    else:
                        S_1_avgs.extend(S_1_avg)

            # append into outputs ############################################
            if not_jtfs or do_S_1_tm:
                for idx, n1 in enumerate(keys1_grouped[k1]):
                    j1 = psi1_f[n1]['j']
                    coef = (S_1[:, idx:idx+1] if vectorized else
                            S_1[idx])
                    out_S_1.append({'coef': coef,
                                    'j': (j1,),
                                    'n': (n1,)})

    # Second order ###########################################################
    if do_S_2 or jtfs_needs_U_2_c:
        if do_S_2:
            cargs = (average, average_global, commons)  # common args

        # execute compute blocks #############################################
        for n2 in U_12_dict:
            U_2_hats = []
            p2f = psi2_f[n2]
            j2 = p2f['j']

            for k1, n_n1s in n_n1s_for_n2_and_k1[n2].items():
                k2 = max(min(j2, log2_T) - k1 - oversampling, 0)

                # convolution + downsampling
                U_1_hats = U_1_hats_grouped[k1][:, :n_n1s]
                if vectorized:
                    U_2_hats.append(_compute_U_2_hat(U_1_hats, k1, k2, p2f, B))
                else:
                    for i in range(U_1_hats.shape[1]):
                        U_1_hat = U_1_hats[:, i:i+1]
                        U_2_hats.append(_compute_U_2_hat(U_1_hat, k1, k2, p2f, B))

            # lowpass filtering / finishing complex convolution ##############
            if vectorized:
                U_2_hat = B.concatenate(U_2_hats, axis=1)
                # Finish the convolution
                U_2_c = B.ifft(U_2_hat)

                if do_S_2:
                    S_2 = _compute_S_2(U_2_c, k1, k2, *cargs)
                else:
                    U_2_cs[n2] = U_2_c
            else:
                if do_S_2:
                    S_2 = []
                else:
                    U_2_cs[n2] = []

                for U_2_hat in U_2_hats:
                    # Finish the convolution
                    U_2_c = B.ifft(U_2_hat)

                    if do_S_2:
                        S_2.append(_compute_S_2(U_2_c, k1, k2, *cargs))
                    else:
                        U_2_cs[n2].append(U_2_c)

            # append into outputs ############################################
            if do_S_2:
                idx = 0
                for n1 in n1s_of_n2[n2]:
                    j1 = psi1_f[n1]['j']
                    coef = (S_2[:, idx:idx+1] if vectorized else
                            S_2[idx])
                    out_S_2.append({'coef': coef,
                                    'j': (j2, j1),
                                    'n': (n2, n1)})
                    idx += 1

    # Pack & return ##########################################################
    if not_jtfs:
        out = out_S_0 + out_S_1 + out_S_2
        if out_type == 'array' and average:
            out = B.concatenate([c['coef'] for c in out])

    else:
        out = {}
        if do_S_0:
            out['S_0'] = S_0
        if do_S_1_tm:
            out['S_1_tms'] = [c['coef'] for c in out_S_1]
        if do_S_1_avg:
            out['S_1_avgs'] = S_1_avgs
        if jtfs_needs_U_2_c:
            out['U_2_cs'] = U_2_cs

    return out


# Helpers: second order ######################################################
def _compute_S_2(U_2_c, k1, k2, average, average_global, commons):
    B, log2_T, _, oversampling, ind_start, ind_end, phi_f = commons

    # Take modulus
    U_2_m = B.modulus(U_2_c)

    # Lowpass filtering
    if average_global:
        S_2 = B.mean(U_2_m, axis=-1)
    elif average:
        U_2_hat = B.r_fft(U_2_m)

        # Convolve with phi_log2_T
        k2_log2_T = max(log2_T - k2 - k1 - oversampling, 0)

        S_2_c = B.multiply(U_2_hat, phi_f[k1 + k2])
        S_2_hat = B.subsample_fourier(S_2_c, 2**k2_log2_T)
        S_2_r = B.ifft_r(S_2_hat)

        S_2 = B.unpad(S_2_r, ind_start[k1 + k2 + k2_log2_T],
                      ind_end[k1 + k2 + k2_log2_T])
    else:
        S_2 = B.unpad(U_2_m, ind_start[k1 + k2], ind_end[k1 + k2])
    return S_2


def _compute_U_2_hat(U_1_hat, k1, k2, p2f, B):
    U_2_c = B.multiply(U_1_hat, p2f[k1])
    U_2_hat = B.subsample_fourier(U_2_c, 2**k2)
    return U_2_hat


# Helpers: first order #######################################################
def _compute_U_1(U_1_arr, vectorized_early_U_1, do_U_1_hat, B, k1=None):
    if vectorized_early_U_1:
        # saved subsampling for doing it all at once
        U_1_c = U_1_arr
        U_1_hat = B.subsample_fourier(U_1_c, 2**k1)
    else:
        # already subsampled
        U_1_hat = U_1_arr

    # finish FFT convolution (back to time domain)
    U_1_c = B.ifft(U_1_hat)
    # Take the modulus
    U_1_m = B.modulus(U_1_c)

    # return, maybe with FFT
    if do_U_1_hat:
        U_1_hat = B.r_fft(U_1_m)
        return U_1_m, U_1_hat
    else:
        return U_1_m


def _convolve_with_phi_first_order(U_1_hat, k1, k1_log2_T, ind_start, ind_end,
                                   phi_f, B):
    # FFT convolution
    S_1_c = B.multiply(U_1_hat, phi_f[k1])
    S_1_hat = B.subsample_fourier(S_1_c, 2**k1_log2_T)
    S_1_r = B.ifft_r(S_1_hat)

    # unpadding
    S_1 = B.unpad(S_1_r, ind_start[k1_log2_T + k1], ind_end[k1_log2_T + k1])
    return S_1


def _lowpass_first_order(k1, U_1_m, U_1_hats_grouped,
                         average, average_global, commons):
    B, log2_T, vectorized, oversampling, ind_start, ind_end, phi_f = commons

    if average_global:
        # Arithmetic mean
        if vectorized:
            # print(U_1_m)  # TODO all zeros from jax test_correctness
            S_1 = B.mean(U_1_m, axis=-1)
        else:
            S_1 = []
            for i in range(len(U_1_m)):
                S_1.append(B.mean(U_1_m[i], axis=-1))

    elif average:
        # Convolve with phi_f
        cargs = (ind_start, ind_end, phi_f, B)  # common args
        k1_log2_T = max(log2_T - k1 - oversampling, 0)
        if vectorized:
            S_1 = _convolve_with_phi_first_order(U_1_hats_grouped[k1], k1,
                                                 k1_log2_T, *cargs)
        else:
            S_1 = []
            for i in range(U_1_hats_grouped[k1].shape[1]):
                S_1.append(_convolve_with_phi_first_order(
                    U_1_hats_grouped[k1][:, i:i+1], k1, k1_log2_T, *cargs))

    else:
        # Unaveraged: simply unpad
        if vectorized:
            S_1 = B.unpad(U_1_m, ind_start[k1], ind_end[k1])
        else:
            S_1 = []
            for i in range(len(U_1_m)):
                S_1.append(B.unpad(U_1_m[i], ind_start[k1], ind_end[k1]))

    return S_1


__all__ = ['scattering1d']
