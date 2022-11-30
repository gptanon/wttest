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
import math
import os
import io
import numpy as np
from wavespin.torch import Scattering1D
from wavespin.numpy import Scattering1D as Scattering1DNumPy
from wavespin.utils.gen_utils import backend_has_gpu
from utils import TEST_DATA_DIR, FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 1

from wavespin.scattering1d.backend.torch_backend import backend as torch_backend
backends = [torch_backend]

if backend_has_gpu('torch'):
    devices = ['cuda', 'cpu']
else:
    devices = ['cpu']

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_simple_scatterings(device, backend, random_state=42):
    """
    Checks the behaviour of the scattering on simple signals
    (zero, constant, pure cosine)

    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_scattering1d.py
    """

    rng = np.random.RandomState(random_state)
    J = 6
    Q = 8
    N = 2**9
    scattering = Scattering1D(N, J, Q, backend=backend).to(device)
    return

    # zero signal
    x0 = torch.zeros(2, N).to(device)

    s = scattering(x0)

    # check that s is zero!
    assert torch.max(torch.abs(s)) < 1e-7

    # constant signal
    x1 = rng.randn(1)[0] * torch.ones(1, N).to(device)

    s1 = scattering(x1)

    # check that all orders above 1 are 0
    assert torch.max(torch.abs(s1[:, 1:])) < 1e-7

    # sinusoid scattering
    meta = scattering.meta()
    for _ in range(3):
        k = rng.randint(1, N // 2, 1)[0]
        x2 = torch.cos(2 * math.pi * float(k) * torch.arange(
            0, N, dtype=torch.float32) / float(N))
        x2 = x2.unsqueeze(0).to(device)

        s2 = scattering(x2)

        assert (s2[:,torch.from_numpy(meta['order']) != 1,:].abs().max()
                < 1e-2)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_sample_scattering(device, backend):
    """
    Applies scattering on a stored signal to make sure its output agrees with
    a previously calculated version.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_scattering1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    with open(os.path.join(TEST_DATA_DIR, 'test_scattering_1d.npz'), 'rb'
              ) as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)


    x = torch.from_numpy(data['x']).to(device)
    J = data['J']
    Q = data['Q']
    Sx0 = torch.from_numpy(data['Sx']).to(device)

    N = x.shape[-1]

    sc = Scattering1D(N, J, Q, backend=backend, max_pad_factor=1,
                      smart_paths='primitive').to(device)

    Sx = sc(x)
    dtype = {'single': 'float32', 'double': 'float64'}[sc.precision]
    Sx0 = Sx0.to(dtype=getattr(torch, dtype))
    assert torch.allclose(Sx, Sx0), "MAE={:.3e}".format(float((Sx - Sx0).mean()))

    # for coverage
    sc.out_type = 'list'
    _ = sc(x)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_computation_Ux(device, backend, random_state=42):
    """
    Checks the computation of the U transform (no averaging for 1st order)

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_scattering1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    rng = np.random.RandomState(random_state)
    J = 6
    Q = 8
    N = 2**12
    scattering = Scattering1D(N, J, Q, average=False, out_type="list",
                              max_order=1, max_pad_factor=1,
                              smart_paths='primitive', backend=backend
                              ).to(device)
    # random signal
    x = torch.from_numpy(rng.randn(1, N)).float().to(device)

    s = scattering(x)

    # check that the keys in s correspond to the order 0 and second order
    sn = [s[i]['n'] for i in range(len(s))]
    for n1 in range(len(scattering.psi1_f)):
        assert (n1,) in sn
    for n in sn:
        if n != ():
            assert n[0] < len(scattering.psi1_f)
        else:
            assert True

    scattering.max_order = 2

    s = scattering(x)

    count = 1
    sn = [s[i]['n'] for i in range(len(s))]
    for n1, p1 in enumerate(scattering.psi1_f):
        assert (n1,) in sn
        count += 1
        for n2, p2 in enumerate(scattering.psi2_f):
            if p2['j'] > p1['j']:
                assert (n2, n1) in sn
                count += 1

    assert count == len(s), (count, len(s))

    with pytest.raises(ValueError) as ve:
        scattering.out_type = "array"
        scattering(x)
    assert "mutually incompatible" in ve.value.args[0]


# Technical tests
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_coordinates(device, backend, random_state=42):
    """
    Tests whether the coordinates correspond to the actual values (obtained
    with Scattering1D.meta()), and with the vectorization

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_scattering1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """

    torch.manual_seed(random_state)
    J = 6
    Q = 8
    N = 2**12

    scattering = Scattering1D(N, J, Q, max_order=2, backend=backend,
                              max_pad_factor=1)

    x = torch.randn(2, N)

    scattering.to(device)
    x = x.to(device)

    for max_order in (1, 2):
        scattering.max_order = max_order

        scattering.out_type = 'list'
        s_dico = scattering(x)
        s_dico = {c['n']: c['coef'].data.cpu() for c in s_dico}

        scattering.out_type = 'array'
        s_vec = scattering(x)
        s_vec = s_vec.cpu()

        meta = scattering.meta()

        assert len(s_dico) == s_vec.shape[1], (
            len(s_dico), s_vec.shape)

        for cc in range(s_vec.shape[1]):
            k = meta['key'][cc]
            assert torch.allclose(s_vec[:, cc], torch.squeeze(s_dico[k]))


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_differentiability_scattering(device, backend, random_state=42):
    """
    It simply tests whether it is really differentiable or not.
    This does NOT test whether the gradients are correct.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_scattering1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    torch.manual_seed(random_state)

    J = 6
    Q = 8
    N = 2**12

    scattering = Scattering1D(N, J, Q, backend=backend, max_pad_factor=1
                              ).to(device)

    x = torch.randn(2, N, requires_grad=True, device=device)

    s = scattering.forward(x)
    loss = torch.sum(torch.abs(s))
    loss.backward()
    assert torch.max(torch.abs(x.grad)) > 0.


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_batch_shape_agnostic(device, backend):
    """
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_scattering1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    J, Q = 3, 8
    length = 1024
    shape = (length,)

    length_ds = length / 2**J

    S = Scattering1D(shape, J, Q, backend=backend).to(device)

    with pytest.raises(ValueError) as ve:
        S(torch.zeros(()).to(device))
    assert "at least 1D" in ve.value.args[0]

    x = torch.zeros(shape).to(device)

    Sx = S(x)

    assert Sx.dim() == 2, Sx.shape
    assert Sx.shape[-1] == length_ds, (Sx.shape, length_ds)

    n_coeffs = Sx.shape[-2]

    test_shapes = ((1,) + shape, (2,) + shape, (2,2) + shape, (2,2,2) + shape)

    for test_shape in test_shapes:
        x = torch.zeros(test_shape).to(device)

        S.out_type = 'array'
        Sx = S(x)

        assert Sx.dim() == len(test_shape)+1
        assert Sx.shape[-1] == length_ds
        assert Sx.shape[-2] == n_coeffs
        assert Sx.shape[:-2] == test_shape[:-1]

        S.out_type = 'list'
        Sx = S(x)

        assert len(Sx) == n_coeffs
        for c in Sx:
            assert c['coef'].shape[-1] == length_ds
            assert c['coef'].shape[:-1] == test_shape[:-1]


@pytest.mark.parametrize("backend", backends)
def test_scattering_GPU_CPU(backend, random_state=42):
    """
    This function tests whether the CPU computations are equivalent to
    the GPU ones

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_scattering1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    if torch.cuda.is_available():
        torch.manual_seed(random_state)

        J = 6
        Q = 8
        N = 2**12

        # build the scattering
        scattering = Scattering1D(N, J, Q, backend=backend, max_pad_factor=1
                                  ).cpu()

        x = torch.randn(2, N)
        s_cpu = scattering(x)

        scattering.gpu()
        x_gpu = x.clone().cuda()
        s_gpu = scattering(x_gpu).cpu()
        # compute the distance

        Warning('Tolerance has been slightly lowered here...')
        assert torch.allclose(s_cpu, s_gpu, atol=1e-7)


@pytest.mark.parametrize("backend", backends)
def test_scattering_shape_input(backend):
    """
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/scattering1d/
    test_torch_scattering1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    # Checks that a wrong input to shape raises an error
    J, Q = 6, 8
    with pytest.raises(ValueError) as ve:
        shape = 5, 6
        _ = Scattering1D(shape, J, Q, backend=backend, max_pad_factor=1)
    assert "exactly one element" in ve.value.args[0]


    with pytest.raises(ValueError) as ve:
        shape = 1.5
        _ = Scattering1D(shape, J, Q, backend=backend, max_pad_factor=1)
        # should invoke the else branch
    assert "1-tuple" in ve.value.args[0]
    assert "integer" in ve.value.args[0]


@pytest.mark.parametrize("backend", backends)
def test_vs_numpy(backend):
    """Test torch's outputs match numpy's within float precision."""
    N = 2048
    Jmax = int(np.log2(N))
    J = Jmax - 2
    Q = 8

    for average in (True, False):
        kw = dict(J=J, Q=Q, shape=N, average=average, max_pad_factor=1,
                  precision='single', out_type='array' if average else 'list')

        ts_torch = Scattering1D(**kw)
        ts_numpy = Scattering1DNumPy(**kw)

        x = np.random.randn(N).astype('float32')
        xt = torch.from_numpy(x)

        out_torch = ts_torch(xt)
        out_numpy = ts_numpy(x)

        if average:
            ae_avg = np.abs(out_torch - out_numpy).mean()
            ae_max = np.abs(out_torch - out_numpy).max()
            assert np.allclose(out_torch, out_numpy), (
                "ae_avg={:.2e}, ae_max={:.2e}".format(ae_avg, ae_max))
        else:
            for i, (ot, on) in enumerate(zip(out_torch, out_numpy)):
                ot, on = ot['coef'].cpu().numpy(), on['coef']
                ae_avg = np.abs(ot - on).mean()
                ae_max = np.abs(ot - on).max()
                assert np.allclose(ot, on, atol=1e-7), (
                    "idx={}, ae_avg={:.2e}, ae_max={:.2e}"
                    ).format(i, ae_avg, ae_max)


@pytest.mark.parametrize("backend", backends)
def test_paths_exclude(backend):
    """Simple test to ensure `paths_exclude` doesn't error in Scattering1D."""
    paths_exclude = {'n2': -1, 'j2': 1}
    N = 1024
    _ = Scattering1D(shape=N, J=7, Q=8, paths_exclude=paths_exclude)


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        for device in devices:
            for backend in backends:
                args = (device, backend)
                test_simple_scatterings(*args)
                test_sample_scattering(*args)
                test_computation_Ux(*args)
                test_coordinates(*args)
                test_differentiability_scattering(*args)
                test_batch_shape_agnostic(*args)
        for backend in backends:
            test_scattering_GPU_CPU(backend)
            test_scattering_shape_input(backend)
            test_vs_numpy(backend)
            test_paths_exclude(backend)
    else:
        pytest.main([__file__, "-s"])
