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
                 average_global=None, vectorized=None, vectorized_early_U_1=None,
                 psi1_f_stacked=None):
    """
    Main function implementing the 1-D scattering transform.
    See `help(wavespin.scattering1d.frontend.Scattering1D)`.

    Computes same coefficients as `wavespin.Scattering1D.scattering1d`, but
    in a different order. Equality is tested in
    `tests/scattering1d/test_legacy_scattering1d.py`.

    References
    ----------
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    frontend/core.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    unpad = backend.unpad
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate
    mean = backend.mean

    # S is simply a dictionary if we do not perform the averaging...
    klog2_T = max(log2_T - oversampling, 0)
    out_S_0, out_S_1, out_S_2 = [], [], []

    # pad to a dyadic size and make it complex
    U_0 = pad_fn(x)
    # compute the Fourier transform
    U_0_hat = rfft(U_0)

    # Zeroth order
    k0 = max(log2_T - oversampling, 0)

    if average_global:
        S_0 = mean(U_0, axis=-1)
    elif average:
        S_0_c = cdgmm(U_0_hat, phi_f[0])
        S_0_hat = subsample_fourier(S_0_c, 2**k0)
        S_0_r = irfft(S_0_hat)

        S_0 = unpad(S_0_r, ind_start[k0], ind_end[k0])
    else:
        S_0 = x
    out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': ()})

    # First order:
    for n1, p1f in enumerate(psi1_f):
        # Convolution + downsampling
        j1 = p1f['j']

        k1 = max(min(j1, log2_T) - oversampling, 0)

        U_1_c = cdgmm(U_0_hat, p1f[0])
        U_1_hat = subsample_fourier(U_1_c, 2**k1)
        U_1_c = ifft(U_1_hat)

        # Take the modulus
        U_1_m = modulus(U_1_c)

        if (average and not average_global) or max_order > 1:
            U_1_hat = rfft(U_1_m)

        if average_global:
            S_1 = mean(U_1_m, axis=-1)
        elif average:
            # Convolve with phi_f
            k1_log2_T = max(log2_T - k1 - oversampling, 0)
            S_1_c = cdgmm(U_1_hat, phi_f[k1])
            S_1_hat = subsample_fourier(S_1_c, 2**k1_log2_T)
            S_1_r = irfft(S_1_hat)

            S_1 = unpad(S_1_r, ind_start[k1_log2_T + k1], ind_end[k1_log2_T + k1])
        else:
            S_1 = unpad(U_1_m, ind_start[k1], ind_end[k1])

        out_S_1.append({'coef': S_1,
                        'j': (j1,),
                        'n': (n1,)})

        if max_order == 2:
            # Second order
            for n2, p2f in enumerate(psi2_f):
                j2 = p2f['j']

                if n1 in paths_include_n2n1[n2]:
                    # Convolution + downsampling
                    k2 = max(min(j2, log2_T) - k1 - oversampling, 0)

                    U_2_c = cdgmm(U_1_hat, p2f[k1])
                    U_2_hat = subsample_fourier(U_2_c, 2**k2)
                    U_2_c = ifft(U_2_hat)

                    # Take the modulus
                    U_2_m = modulus(U_2_c)

                    if average_global:
                        S_2 = mean(U_2_m, axis=-1)
                    elif average:
                        U_2_hat = rfft(U_2_m)

                        # Convolve with phi_f
                        k2_log2_T = max(log2_T - k2 - k1 - oversampling, 0)

                        S_2_c = cdgmm(U_2_hat, phi_f[k1 + k2])
                        S_2_hat = subsample_fourier(S_2_c, 2**k2_log2_T)
                        S_2_r = irfft(S_2_hat)

                        S_2 = unpad(S_2_r, ind_start[k1 + k2 + k2_log2_T],
                                    ind_end[k1 + k2 + k2_log2_T])
                    else:
                        S_2 = unpad(U_2_m, ind_start[k1 + k2], ind_end[k1 + k2])

                    out_S_2.append({'coef': S_2,
                                    'j': (j1, j2),
                                    'n': (n1, n2)})

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    if out_type == 'array' and average:
        out_S = concatenate([c['coef'] for c in out_S])

    return out_S


__all__ = ['scattering1d']
