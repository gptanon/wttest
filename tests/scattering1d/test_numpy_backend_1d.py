# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018- The Kymatio developers
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the Modified BSD License
# (BSD 3-clause; see NOTICE.txt in the WaveSpin root directory for details).
# -----------------------------------------------------------------------------
import pytest
import numpy as np

from wavespin.scattering1d.backend.numpy_backend import backend
from utils import FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 0


def test_subsample_fourier():
    """
    This is a modification of `tests/scattering1d/test_numpy_backend_1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    J = 10
    # 1d signal
    x = np.random.randn(2, 2 ** J) + 1j * np.random.randn(2, 2 ** J)
    x_f = np.fft.fft(x, axis=-1)

    for j in range(J + 1):
        x_f_sub = backend.subsample_fourier(x_f, 2 ** j)
        x_sub = np.fft.ifft(x_f_sub, axis=-1)
        assert np.allclose(x[:, ::2 ** j], x_sub)


def test_pad():
    """
    From `tests/scattering1d/test_numpy_backend_1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    N = 128
    x = np.random.rand(2, 4, N)

    for pad_left in range(0, N - 16, 16):
        for pad_right in [pad_left, pad_left + 16]:
            x_pad = backend.pad(x, pad_left, pad_right)

            # compare left reflected part of padded array with left side
            # of original array
            for t in range(1, pad_left + 1):
                assert np.allclose(x_pad[..., pad_left - t], x[..., t])
            # compare left part of padded array with left side of
            # original array
            for t in range(x.shape[-1]):
                assert np.allclose(x_pad[..., pad_left + t], x[..., t])
            # compare right reflected part of padded array with right side
            # of original array
            for t in range(1, pad_right + 1):
                assert np.allclose(x_pad[..., x_pad.shape[-1] - 1 -
                                         pad_right + t],
                                   x[..., x.shape[-1] - 1 - t])
            # compare right part of padded array with right side of
            # original array
            for t in range(1, pad_right + 1):
                assert np.allclose(x_pad[..., x_pad.shape[-1] - 1 -
                                         pad_right - t],
                                   x[..., x.shape[-1] - 1 - t])


def test_unpad():
    """
    From `tests/scattering1d/test_numpy_backend_1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    # test unpading of a random tensor
    x = np.random.rand(8, 4)

    y = backend.unpad(x, 1, 3)

    assert y.shape == (8, 2)
    assert np.allclose(y, x[:, 1:3])

    N = 128
    x = np.random.rand(2, 4, N)

    # similar to for loop in pad test
    for pad_left in range(0, N - 16, 16):
        pad_right = pad_left + 16
        x_pad = backend.pad(x, pad_left, pad_right)
        x_unpadded = backend.unpad(x_pad, pad_left, x_pad.shape[-1] - pad_right)
        assert np.allclose(x, x_unpadded)


def test_fft_type():
    """NumPy doesn't care!"""
    pass


def test_fft():
    """
    From `tests/scattering1d/test_numpy_backend_1d.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def coefficent(n):
        return np.exp(-2 * np.pi * 1j * n)

    x_r = np.random.rand(4)

    I, K = np.meshgrid(np.arange(4), np.arange(4), indexing='ij')

    coefficents = coefficent(K * I / x_r.shape[0])

    y_r = (x_r * coefficents).sum(-1)

    z = backend.r_fft(x_r)
    assert np.allclose(y_r, z)

    z_1 = backend.ifft(z)
    assert np.allclose(x_r, z_1)

    z_2 = backend.ifft_r(z)
    assert not np.iscomplexobj(z_2)
    assert np.allclose(x_r, z_2)


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_subsample_fourier()
        test_pad()
        test_unpad()
        test_fft_type()
        test_fft()
    else:
        pytest.main([__file__, "-s"])
