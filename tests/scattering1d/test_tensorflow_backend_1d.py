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

# skip this test if no TF installed
from utils import cant_import, FORCED_PYTEST
got_tf = bool(not cant_import('tensorflow'))
if got_tf:
    from wavespin.scattering1d.backend.tensorflow_backend import backend


# set True to execute all test functions without pytest
run_without_pytest = 0


def test_subsample_fourier():
    """
    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_tensorflow_backend_1d.py
    """
    if not got_tf:
        return None if run_without_pytest else pytest.skip()
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
    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_tensorflow_backend_1d.py
    """
    if not got_tf:
        return None if run_without_pytest else pytest.skip()
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
                assert np.allclose(
                    x_pad[..., x_pad.shape[-1] - 1 - pad_right + t],
                    x[..., x.shape[-1] - 1 - t])
            # compare right part of padded array with right side of
            # original array
            for t in range(1, pad_right + 1):
                assert np.allclose(
                    x_pad[..., x_pad.shape[-1] - 1 - pad_right - t],
                    x[..., x.shape[-1] - 1 - t])


def test_unpad():
    """
    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_tensorflow_backend_1d.py
    """
    if not got_tf:
        return None if run_without_pytest else pytest.skip()
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
    """
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_tensorflow_backend_1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    if not got_tf:
        return None if run_without_pytest else pytest.skip()
    x = np.random.rand(8, 4) + 1j * np.random.rand(8, 4)

    with pytest.raises(TypeError) as record:
        _ = backend.r_fft(x)
    assert 'should be real' in record.value.args[0]

    x = np.random.rand(8, 4)

    with pytest.raises(TypeError) as record:
        _ = backend.ifft(x)
    assert 'should be complex' in record.value.args[0]

    with pytest.raises(TypeError) as record:
        _ = backend.ifft_r(x)
    assert 'should be complex' in record.value.args[0]


def test_fft():
    """
    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_tensorflow_backend_1d.py
    """
    if not got_tf:
        return None if run_without_pytest else pytest.skip()
    def coefficent(n):
        return np.exp(-2 * np.pi * 1j * n)

    x_r = np.random.rand(4)

    I, K = np.meshgrid(np.arange(4), np.arange(4), indexing='ij')

    coefficents = coefficent(K * I / x_r.shape[0])

    y_r = (x_r * coefficents).sum(-1)

    z = backend.r_fft(x_r)
    # increase tolerance here as tensorflow fft is slightly inaccurate due to
    # eigen implementation https://github.com/google/jax/issues/2952
    # (see also below)
    assert np.allclose(y_r, z, atol=1e-6, rtol=1e-7)

    z_1 = backend.ifft(z)
    assert np.allclose(x_r, z_1, atol=1e-6, rtol=1e-7)

    z_2 = backend.ifft_r(z)
    assert not np.iscomplexobj(z_2)
    assert np.allclose(x_r, z_2, atol=1e-6, rtol=1e-7)


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_subsample_fourier()
        test_pad()
        test_unpad()
        test_fft_type()
        test_fft()
    else:
        pytest.main([__file__, "-s"])
