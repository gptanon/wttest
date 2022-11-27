# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018- The Kymatio developers
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the Modified BSD License
# (BSD 3-clause; see NOTICE.txt in the WaveSpin root directory for details).
# -----------------------------------------------------------------------------
import pytest
import torch
import numpy as np
from utils import FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 0

backends = []

from wavespin.scattering1d.backend.torch_backend import backend
backends.append(backend)
del backend

if torch.cuda.is_available():
    devices = ['cuda', 'cpu']
else:
    devices = ['cpu']


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_pad_1d(device, backend, random_state=42):
    """
    Tests the correctness and differentiability of pad_1d

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_backend_1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    torch.manual_seed(random_state)
    N = 128
    for pad_left in range(0, N - 16, 16):
        for pad_right in [pad_left, pad_left + 16]:
            x = torch.randn(2, 4, N, requires_grad=True, device=device)
            x_pad = backend.pad(x, pad_left, pad_right)
            x_pad = x_pad.reshape(x_pad.shape)
            # Check the size
            x2 = x.clone()
            x_pad2 = x_pad.clone()
            # compare left reflected part of padded array with left side
            # of original array
            for t in range(1, pad_left + 1):
                assert torch.allclose(x_pad2[..., pad_left - t], x2[..., t])
            # compare left part of padded array with left side of
            # original array
            for t in range(x.shape[-1]):
                assert torch.allclose(x_pad2[..., pad_left + t], x2[..., t])
            # compare right reflected part of padded array with right side
            # of original array
            for t in range(1, pad_right + 1):
                assert torch.allclose(
                    x_pad2[..., x_pad.shape[-1] - 1 - pad_right + t],
                    x2[..., x.shape[-1] - 1 - t])
            # compare right part of padded array with right side of
            # original array
            for t in range(1, pad_right + 1):
                assert torch.allclose(
                    x_pad2[..., x_pad.shape[-1] - 1 - pad_right - t],
                    x2[..., x.shape[-1] - 1 - t])

            # check the differentiability
            loss = 0.5 * torch.sum(x_pad**2)
            loss.backward()
            # compute the theoretical gradient for x
            x_grad_original = x.clone()
            x_grad = x_grad_original.new(x_grad_original.shape).fill_(0.)
            x_grad += x_grad_original

            for t in range(1, pad_left + 1):
                x_grad[..., t] += x_grad_original[..., t]

            for t in range(1, pad_right + 1):  # it is counted twice!
                t0 = x.shape[-1] - 1 - t
                x_grad[..., t0] += x_grad_original[..., t0]

            # get the difference
            assert torch.allclose(x.grad, x_grad)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_modulus(device, backend, random_state=42):
    """
    Tests the stability and differentiability of modulus.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_backend_1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    torch.manual_seed(random_state)
    # Test with a random vector
    x = torch.randn(2, 4, 128, 2, requires_grad=True, device=device,
                    dtype=torch.complex128)

    x_abs = backend.modulus(x)
    assert x_abs.ndim == x.ndim

    # check the value
    x_abs2 = x_abs.clone()
    x2 = x.clone()
    assert torch.allclose(x_abs2, torch.sqrt(x2.real ** 2 + x2.imag ** 2))

    # check the gradient
    loss = torch.sum(x_abs)
    loss.backward()
    x_grad = x2 / x_abs2
    assert torch.allclose(x.grad, x_grad)


    # Test the differentiation with a vector made of zeros
    x0 = torch.zeros(100, 4, 128, 2, requires_grad=True, device=device,
                     dtype=torch.complex128)
    x_abs0 = backend.modulus(x0)
    loss0 = torch.sum(x_abs0)
    loss0.backward()
    assert torch.max(torch.abs(x0.grad)) <= 1e-7


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_modulus_general(device, backend, random_state=42):
    """
    Tests the stability and differentiability of modulus, general case

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/general/
    test_torch_backend.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    x = torch.randn(1000, 40, 128, dtype=torch.complex128, requires_grad=True)
    xd = x.detach()
    xv = torch.view_as_real(xd)
    x_abs = torch.abs(xd)

    xv_grad = xv.clone()
    xv_grad[..., 0] = xv[..., 0] / x_abs
    xv_grad[..., 1] = xv[..., 1] / x_abs
    xv_grad = torch.view_as_complex(xv_grad)

    y = torch.abs(x)
    y_grad = torch.ones_like(y)
    y.backward(gradient=y_grad)
    x_grad_manual = x.grad
    assert torch.allclose(x_grad_manual, xv_grad)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_subsample_fourier(device, backend, random_state=42):
    """
    Tests whether the periodization in Fourier performs a good subsampling
    in time.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_backend_1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    rng = np.random.RandomState(random_state)
    J = 10
    x = rng.randn(2, 4, 2**J) + 1j * rng.randn(2, 4, 2**J)
    x_f = np.fft.fft(x, axis=-1)
    x_f_th = torch.from_numpy(x_f).to(device)

    for j in range(J + 1):
        x_f_sub_th = backend.subsample_fourier(x_f_th, 2**j).cpu()
        x_f_sub = x_f_sub_th.numpy()
        x_f_sub.dtype = 'complex128'
        x_sub = np.fft.ifft(x_f_sub, axis=-1)
        assert np.allclose(x[:, :, ::2**j], x_sub)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_unpad(device, backend):
    """
    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_backend_1d.py
    """
    # test unpading of a random tensor
    x = torch.randn(8, 4).to(device)

    y = backend.unpad(x, 1, 3)

    assert y.shape == (8, 2)
    assert torch.allclose(y, x[:, 1:3])

    N = 128
    x = torch.rand(2, 4, N).to(device)

    # similar to for loop in pad test
    for pad_left in range(0, N - 16, 16):
        pad_right = pad_left + 16
        x_pad = backend.pad(x, pad_left, pad_right)
        x_unpadded = backend.unpad(x_pad, pad_left, x_pad.shape[-1] - pad_right)
        assert torch.allclose(x, x_unpadded)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_fft_type(device, backend):
    """Torch doesn't care!"""
    pass


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_fft(device, backend):
    """
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_backend_1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def coefficent(n):
        return np.exp(-2 * np.pi * 1j * n)

    x_r = np.random.rand(4)

    I, K = np.meshgrid(np.arange(4), np.arange(4), indexing='ij')

    coefficents = coefficent(K * I / x_r.shape[0])

    y_r = (x_r * coefficents).sum(-1)

    x_r = torch.from_numpy(x_r).to(device)
    y_r = torch.from_numpy(np.column_stack((y_r.real, y_r.imag))).to(device)

    z = backend.r_fft(x_r)
    assert torch.allclose(y_r[..., 0], z.real)

    z_1 = backend.ifft(z)
    assert torch.allclose(x_r, z_1.real)

    z_2 = backend.ifft_r(z)
    assert torch.allclose(x_r, z_2)


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        for device in devices:
            for backend in backends:
                args = (device, backend)
                test_pad_1d(*args)
                test_modulus(*args)
                test_modulus_general(*args)
                test_subsample_fourier(*args)
                test_unpad(*args)
                test_fft_type(*args)
                test_fft(*args)
    else:
        pytest.main([__file__, "-s"])
