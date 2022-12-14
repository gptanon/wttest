# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------

def cwt1d(x, hop_size, pad_fn, backend, psi1_f, psi1_f_stacked,
          cwt_unpad_indices, vectorized):
    """
    Main function implementing the 1-D Continuous Wavelet Transform.
    See `help(wavespin.scattering1d.frontend.Scattering1D)`.
    """
    B = backend
    ind_start, ind_end = cwt_unpad_indices[hop_size]

    # pad & go to Fourier
    xp = pad_fn(x)
    xpf = B.r_fft(xp)

    # FFT-convolve, subsample, unpad
    if vectorized:
        # compute all at once
        c = B.multiply(xpf, psi1_f_stacked)
        c = B.subsample_fourier(c, hop_size)
        c = B.ifft(c)
        out = B.unpad(c, ind_start, ind_end)
    else:
        out = []
        for i, pf in enumerate(psi1_f):
            c = B.multiply(xpf, pf[0])
            c = B.subsample_fourier(c, hop_size)
            c = B.ifft(c)
            c = B.unpad(c, ind_start, ind_end)
            out.append(c)
        out = B.concatenate(out, axis=1)

    # postprocess, return
    out = B.try_squeeze(out, axis=0)
    return out


__all__ = ['cwt1d']
