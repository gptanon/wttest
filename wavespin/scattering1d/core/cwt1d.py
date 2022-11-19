# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------

def cwt1d(x, hop_size, pad_fn, backend, psi1_f, psi1_f_stacked,
          cwt_unpad_indices, vectorized, squeeze_batch_dim):
    """
    Main function implementing the 1-D Continuous Wavelet Transform.
    See `help(wavespin.scattering1d.frontend.Scattering1D)`.
    """
    B = backend
    ind_start, ind_end = cwt_unpad_indices[hop_size]

    # pad & go to Fourier
    xp = pad_fn(x)
    xpf = B.rfft(xp)

    # FFT-convolve, subsample, unpad
    if vectorized:
        # compute all at once
        c = B.cdgmm(xpf, psi1_f_stacked)
        c = B.subsample_fourier(c, hop_size)
        c = B.ifft(c)
        out = B.unpad(c, ind_start, ind_end)
    else:
        n_coeffs = len(psi1_f)
        n_time = ind_end - ind_start
        out = B.zeros_like(xp, shape=(n_coeffs, n_time))
        for i, pf in enumerate(psi1_f):
            c = B.cdgmm(xpf, pf)
            c = B.subsample_fourier(c, hop_size)
            c = B.ifft(c)
            out[i] = B.unpad(c, ind_start, ind_end)

    # postprocess, return
    if squeeze_batch_dim:
        out = B.try_squeeze(out, axis=0)
    return out


__all__ = ['cwt1d']
