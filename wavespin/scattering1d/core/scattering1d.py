# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------


def scattering1d(x, pad_fn, backend, log2_T, psi1_f, psi2_f, phi_f,
                 paths_include_n2n1, ind_start=None, ind_end=None,
                 oversampling=0, max_order=2, average=True, out_type='array',
                 average_global=None, vectorized=True, vectorized_early_U_1=None,
                 psi1_f_stacked=None):
    """
    Main function implementing the 1-D scattering transform.
    See `help(wavespin.scattering1d.frontend.Scattering1D)`.

    For an easier to understand equivalent implementation (that's slower), see
    `tests/scattering1d/funcs/scattering1d_legacy.py`.
    """
    B = backend
    out_S_0, out_S_1, out_S_2 = [], [], []

    # pad to a dyadic size and make it complex
    U_0 = pad_fn(x)
    # compute the Fourier transform
    U_0_hat = B.rfft(U_0)

    # Zeroth order ###########################################################
    k0 = max(log2_T - oversampling, 0)

    if average_global:
        S_0 = B.mean(U_0, axis=-1)
    elif average:
        S_0_c = B.cdgmm(U_0_hat, phi_f[0])
        S_0_hat = B.subsample_fourier(S_0_c, 2**k0)
        S_0_r = B.irfft(S_0_hat)

        S_0 = B.unpad(S_0_r, ind_start[k0], ind_end[k0])
    else:
        S_0 = x
    out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': ()})

    # First order ############################################################
    # make compute blocks ####################################################
    U_1_dict = {}
    for n1, p1f in enumerate(psi1_f):
        j1 = p1f['j']
        k1 = max(min(j1, log2_T) - oversampling, 0)
        if k1 not in U_1_dict:
            U_1_dict[k1] = 0
        U_1_dict[k1] += 1
    # preallocate
    U_1_hats_grouped = {}
    for k1 in U_1_dict:
        shape = list(U_0_hat.shape)
        shape[1] = U_1_dict[k1]
        shape[-1] //= 2**k1
        U_1_hats_grouped[k1] = B.zeros_like(U_0_hat, shape)

    # clear namespace to not accidentally reuse from wrong loop
    del n1, p1f, k1

    # execute compute blocks #################################################
    if vectorized_early_U_1:
        # multiply `fft(x)` by all filters at once
        U_1_cs_grouped = B.cdgmm(U_0_hat, psi1_f_stacked)

    keys1 = []
    keys1_grouped = {}
    for n1, p1f in enumerate(psi1_f):
        # Convolution + downsampling
        j1 = p1f['j']
        k1 = max(min(j1, log2_T) - oversampling, 0)

        if not vectorized_early_U_1:
            U_1_c = B.cdgmm(U_0_hat, p1f[0])
            U_1_hat = B.subsample_fourier(U_1_c, 2**k1)

        # Store coefficient in proper grouping
        offset, k = 0, 0
        while k < k1:
            offset += U_1_dict[k]
            k += 1
        if not vectorized_early_U_1:
            U_1_hats_grouped[k1][:, n1 - offset] = U_1_hat
        keys1.append((k1, n1))
        if k1 not in keys1_grouped:
            keys1_grouped[k1] = []
        keys1_grouped[k1].append(n1)

    # lowpass filtering ######################################################
    do_U_1_hat = bool((average and not average_global) or max_order > 1)

    def compute_U_1_hat(U_1_arr, k1=None):
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
            U_1_hat = B.rfft(U_1_m)
            return U_1_m, U_1_hat
        else:
            return U_1_m

    def convolve_with_phi(U_1_hat, k1, k1_log2_T):
        # FFT convolution
        S_1_c = B.cdgmm(U_1_hat, phi_f[k1])
        S_1_hat = B.subsample_fourier(S_1_c, 2**k1_log2_T)
        S_1_r = B.irfft(S_1_hat)

        # unpadding
        S_1 = B.unpad(S_1_r, ind_start[k1_log2_T + k1], ind_end[k1_log2_T + k1])
        return S_1

    for k1 in U_1_hats_grouped:
        # group computation by lengths, controlled by `k1`
        if vectorized:
            # slice coeffs with same subsampled length, controlled by `k1`
            if vectorized_early_U_1:
                start, end = keys1_grouped[k1][0], keys1_grouped[k1][-1] + 1
                U_1_arr = U_1_cs_grouped[:, start:end]
                out = compute_U_1_hat(U_1_arr, k1)
            else:
                U_1_arr = U_1_hats_grouped[k1]
                out = compute_U_1_hat(U_1_arr)

            if do_U_1_hat:
                U_1_m, U_1_hat = out
                U_1_hats_grouped[k1] = U_1_hat
            else:
                U_1_m = out

        else:
            U_1_arr = U_1_hats_grouped[k1]
            U_1_m = []
            for i in range(U_1_arr.shape[1]):
                out = compute_U_1_hat(U_1_arr[:, i:i+1])
                if do_U_1_hat:
                    U_1_m.append(out[0])
                    U_1_hats_grouped[k1][:, i:i+1] = out[1]
                else:
                    U_1_m.append(out)

        # Lowpass filtering
        if average_global:
            # Arithmetic mean
            if vectorized:
                S_1 = B.mean(U_1_m, axis=-1)
            else:
                S_1 = []
                for i in range(len(U_1_m)):
                    S_1.append(B.mean(U_1_m[i], axis=-1))

        elif average:
            # Convolve with phi_f
            k1_log2_T = max(log2_T - k1 - oversampling, 0)
            if vectorized:
                S_1 = convolve_with_phi(U_1_hat, k1, k1_log2_T)
            else:
                S_1 = []
                for i in range(U_1_hats_grouped[k1].shape[1]):
                    S_1.append(convolve_with_phi(
                        U_1_hats_grouped[k1][:, i:i+1], k1, k1_log2_T))

        else:
            # Unaveraged: simply unpad
            if vectorized:
                S_1 = B.unpad(U_1_m, ind_start[k1], ind_end[k1])
            else:
                S_1 = []
                for i in range(len(U_1_m)):
                    S_1.append(B.unpad(U_1_m[i], ind_start[k1], ind_end[k1]))

        # append into outputs ############################################
        for idx, n1 in enumerate(keys1_grouped[k1]):
            j1 = psi1_f[n1]['j']
            coef = (S_1[:, idx:idx+1] if vectorized else
                    S_1[idx])
            out_S_1.append({'coef': coef,
                            'j': (j1,),
                            'n': (n1,)})

    # Second order ###########################################################
    if max_order == 2:
        # make compute blocks ################################################
        U_12_dict = {}
        # here we just append metadata for later use: which n2 will be realized,
        # and their corresponding n1, grouped by k1
        for n2, p2f in enumerate(psi2_f):
            j2 = p2f['j']

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

        # clear namespace to not accidentally reuse from wrong loop
        del n2, p2f, j1, j2, k1, n1, _n1

        # define helpers #####################################################
        def compute_S_2(U_2_hat, k1, k2):
            # Finish the convolution
            U_2_c = B.ifft(U_2_hat)
            # Take modulus
            U_2_m = B.modulus(U_2_c)

            # Lowpass filtering
            if average_global:
                S_2 = B.mean(U_2_m, axis=-1)
            elif average:
                U_2_hat = B.rfft(U_2_m)

                # Convolve with phi_log2_T
                k2_log2_T = max(log2_T - k2 - k1 - oversampling, 0)

                S_2_c = B.cdgmm(U_2_hat, phi_f[k1 + k2])
                S_2_hat = B.subsample_fourier(S_2_c, 2**k2_log2_T)
                S_2_r = B.irfft(S_2_hat)

                S_2 = B.unpad(S_2_r, ind_start[k1 + k2 + k2_log2_T],
                              ind_end[k1 + k2 + k2_log2_T])
            else:
                S_2 = B.unpad(U_2_m, ind_start[k1 + k2], ind_end[k1 + k2])
            return S_2

        def compute_U_2_hat(U_1_hat, k1, k2):
            U_2_c = B.cdgmm(U_1_hat, p2f[k1])
            U_2_hat = B.subsample_fourier(U_2_c, 2**k2)
            return U_2_hat

        # execute compute blocks #############################################
        for n2 in U_12_dict:
            U_2_hats = []
            keys2 = []
            p2f = psi2_f[n2]
            j2 = p2f['j']

            for k1 in U_12_dict[n2]:
                k2 = max(min(j2, log2_T) - k1 - oversampling, 0)

                # convolution + downsampling
                U_1_hats = U_1_hats_grouped[k1]
                if vectorized:
                    U_2_hats.append(compute_U_2_hat(U_1_hats, k1, k2))
                else:
                    for i in range(U_1_hats.shape[1]):
                        U_1_hat = U_1_hats[:, i:i+1]
                        U_2_hats.append(compute_U_2_hat(U_1_hat, k1, k2))

                # Used for sanity check that the right n2-n1 were computed
                keys2.extend(U_12_dict[n2][k1])

            # lowpass filtering ##############################################
            if vectorized:
                U_2_hat = B.concatenate_v2(U_2_hats, axis=1)
                S_2 = compute_S_2(U_2_hat, k1, k2)
            else:
                S_2 = []
                for U_2_hat in U_2_hats:
                    S_2.append(compute_S_2(U_2_hat, k1, k2))

            # append into outputs ############################################
            idx = 0
            for n1, p1f in enumerate(psi1_f):
                if n1 not in paths_include_n2n1[n2]:
                    continue
                assert n1 == keys2[idx], (n1, keys2[idx], idx)

                j1 = p1f['j']
                coef = (S_2[:, idx:idx+1] if vectorized else
                        S_2[idx])
                out_S_2.append({'coef': coef,
                                'j': (j2, j1),
                                'n': (n2, n1)})
                idx += 1

    # Pack & return ##########################################################
    out_S = out_S_0 + out_S_1 + out_S_2

    if out_type == 'array' and average:
        out_S = B.concatenate([x['coef'] for x in out_S])

    return out_S


__all__ = ['scattering1d']
