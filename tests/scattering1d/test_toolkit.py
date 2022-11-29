# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""wavespin/toolkit.py tests. Most methods are tested in test_jtfs.py."""
import pytest
import warnings
import numpy as np
import scipy.signal

from wavespin.toolkit import normalize, tensor_padded, validate_filterbank
from wavespin.utils.gen_utils import ExtendedUnifiedBackend, npy, backend_has_gpu
from wavespin import toolkit as tkt
from wavespin import Scattering1D, TimeFrequencyScattering1D
from utils import tempdir, cant_import, SKIPS, FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 1
# set True to disable matplotlib plots
# (done automatically for CI via `conftest.py`, but `False` here takes precedence)
no_plots = 1
# set True to skip this file entirely
SKIP_ALL = SKIPS['toolkit']

# set backends based on what's available
backends = ['numpy']
if not cant_import('torch'):
    backends.append('torch')
    import torch
if not cant_import('tensorflow'):
    backends.append('tensorflow')
    import tensorflow as tf
if not cant_import('jax'):
    import jax
    backends.append('jax')

#### Tests ###################################################################

@pytest.mark.parametrize("backend", backends)
def test_normalize(backend):
    """Ensure error thrown upon invalid input, but otherwise method
    doesn't error.
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    for mean_axis in (0, (1, 2), -1):
      for std_axis in (0, (1, 2), -1):
        for C in (None, 2):
          for mu in (None, 2):
            for dim0 in (1, 64):
              for dim1 in (1, 65):
                for dim2 in (1, 66):
                    if dim0 == dim1 == dim2 == 1:
                        # invalid combo
                        continue
                    # no `abs` for coverage
                    if backend == 'numpy':
                        x = np.random.randn(dim0, dim1, dim2)
                    elif backend == 'torch':
                        x = torch.randn(dim0, dim1, dim2)
                    elif backend == 'tensorflow':
                        x = tf.random.normal((dim0, dim1, dim2))
                    elif backend == 'jax':
                        x = jax.random.normal(jax.random.PRNGKey(0),
                                              (dim0, dim1, dim2))

                    test_params = dict(
                        mean_axis=mean_axis, std_axis=std_axis, C=C, mu=mu,
                        dim0=dim0, dim1=dim1, dim2=dim2)
                    test_params_str = '\n'.join(f'{k}={v}' for k, v in
                                                test_params.items())
                    try:
                        kw = {k: v for k, v in test_params.items()
                              if k not in ('dim0', 'dim1', 'dim2')}
                        _ = normalize(x, **kw)
                    except ValueError as e:
                        if "input dims cannot be" in str(e):
                            continue
                        elif (backend == 'tensorflow' and mu is None and
                              "mu=None" in str(e)):
                            continue
                        else:
                            print(test_params_str)
                            raise e
                    except Exception as e:
                        print(test_params_str)
                        raise e


@pytest.mark.parametrize("backend", backends)
def test_tensor_padded(backend):
    """Test `tensor_padded` works as intended."""
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    # helper methods #########################################################
    if backend == 'numpy':
        tensor_fn = np.array
    elif backend == 'torch':
        tensor_fn = torch.tensor
    elif backend == 'tensorflow':
        tensor_fn = tf.constant
    elif backend == 'jax':
        tensor_fn = jax.numpy.array

    def tensor(x):
        if len(x[0]) != len(x[1]):  # ragged case
            if backend == 'numpy':
                return x
            return [[tensor_fn(dim1) for dim1 in dim0] for dim0 in x]
        return tensor_fn(x)

    def assert_fn(target, out):
        assert np.allclose(npy(target), npy(out)), "{}\n{}".format(backend, out)

    # test ###################################################################

    seq = tensor(
        [[[1, 2, 3, 4],
          [1, 2, 3],],
         [[1, 2, 3],
          [1, 2],
          [1],],
         ]
    )
    # right-pad case
    target = tensor_fn([[[1, 2, 3, 4],
                         [1, 2, 3, 0],
                         [0, 0, 0, 0]],
                        [[1, 2, 3, 0],
                         [1, 2, 0, 0],
                         [1, 0, 0, 0]]])
    out = tensor_padded(seq)
    assert_fn(target, out)

    # left-pad case
    target = tensor_fn([[[1, 2, 3, 4],
                         [0, 1, 2, 3],
                         [0, 0, 0, 0]],
                        [[0, 1, 2, 3],
                         [0, 0, 1, 2],
                         [0, 0, 0, 1]]])
    out = tensor_padded(seq, left_pad_axis=-1)
    assert_fn(target, out)

    # with `pad_value`
    # write manually because tensorflow
    target = tensor_fn([[[1,   2,  3,  4],
                         [-2,  1,  2,  3],
                         [-2, -2, -2, -2]],
                        [[-2,  1,  2,  3],
                         [-2, -2,  1,  2],
                         [-2, -2, -2,  1]]])
    out = tensor_padded(seq, left_pad_axis=-1, pad_value=-2)
    assert_fn(target, out)


def test_validate_filterbank():
    """Try to include every printable report status."""
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    with warnings.catch_warnings(record=True) as _:
        p0 = np.hstack([np.ones(28), np.zeros(100)]) + 0.
        p1 = p0.copy()[::-1]
        p2 = p0.copy()[18:] + 1j
        psi_fs = [p0, p1, p2] * 2
        _ = validate_filterbank(psi_fs, is_time_domain=0)

        psi_fs = [p1 + p0] * 6
        _ = validate_filterbank(psi_fs, is_time_domain=0)
        _ = validate_filterbank(psi_fs, is_time_domain=1)


def test_fit_smart_paths():
    class MyGen():
        def __init__(self, N):
            self.X_all = np.random.randn(20, N)

        def __getitem__(self, idx):
            if idx >= len(self):  # needed if method doesn't throw IndexError
                raise IndexError  # so here not needed per `X_all[idx]`
            return self.X_all[idx]

        def __len__(self):
            return len(self.X_all)

    N = 256
    x_all = MyGen(N)

    jtfs = TimeFrequencyScattering1D(N)
    sc = Scattering1D(**{k: getattr(jtfs, k) for k in
                         ('shape', 'J', 'Q', 'T', 'max_pad_factor')})

    # standard usage
    tkt.fit_smart_paths(sc, x_all)

    # using `outs_dir` and non-default `e_loss_goal`
    with tempdir() as outs_dir:
        tkt._compute_e_fulls(sc, x_all, outs_dir=outs_dir)
        tkt.fit_smart_paths(sc, x_all, outs_dir=outs_dir, e_loss_goal=0.0314159)


def test_decimate():
    """Tests that `Decimate`
        - outputs match that of `scipy.signal.decimate` for different
        axes, `x.ndim`, and backends.
        - is differentiable for 'torch' backend, on CPU and GPU

    Also test that `F_kind='decimate'` doesn't error in JTFS.
    """
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()

    def decimate0(x, q, axis=-1):
        x = npy(x)
        return scipy.signal.decimate(x, q, axis=axis, ftype='fir')

    def decimate1(d_obj, x, q, axis=-1, backend='numpy'):
        return d_obj(x, q, axis=axis)

    ndim = 2
    for backend in ('numpy', 'torch', 'jax'):
        if cant_import(backend):
            continue
        elif backend == 'torch':
            import torch
        elif backend == 'jax':
            import jax

        # make here so we can test device-switching
        d_obj = tkt.Decimate(backend=backend, sign_correction=None,
                             dtype='float32')

        for N_scale in range(1, 10):
            if N_scale > 7 and backend == 'jax':
                continue  # too slow locally...

            # make `x` -------------------------------------------------------
            # also test non-dyadic length
            if N_scale == 9:
                N = int(2**9.4)
            else:
                N = 2**N_scale
            shape = [N, max(N//2, 1), max(N//4, 1)][:ndim]

            np.random.seed(0)
            x = np.random.randn(*shape)
            if backend == 'torch':
                x = torch.from_numpy(x)
            elif backend == 'jax':
                x = jax.numpy.array(x)

            # devices, if multiple available for backend ---------------------
            devices = ['cpu']
            if backend_has_gpu(backend):
                devices.append('gpu')

            # test -----------------------------------------------------------
            for device in devices:
                if device == 'gpu':
                    if backend == 'torch':
                        x = x.cuda()
                    elif backend == 'jax':
                        x = jax.device_put(x, device=jax.devices('gpu')[0])

                for factor_scale in range(1, N_scale + 1):
                    factor = 2**factor_scale

                    for axis in range(x.ndim):
                        o0 = decimate0(x, factor, axis)

                        # test + and - versions of `axis`
                        for ax in (axis, axis - x.ndim):
                            o1 = decimate1(d_obj, x, factor, ax, backend)

                            # cause float32, but unsure why jax is worse
                            atol = 1e-7 if backend != 'jax' else 1e-6

                            o1 = npy(o1)
                            assert np.allclose(o0, o1, atol=atol), (
                                backend, device, N_scale, factor_scale, ax)

    # test torch differentiability
    if not cant_import('torch'):
        for device in ('cpu', 'cuda'):
            if device == 'cuda' and not backend_has_gpu('torch'):
                continue
            x = torch.randn(4).to(device)
            x.requires_grad = True

            o = tkt.Decimate(backend='torch', gpu=(device=='cuda'))(x, 2)
            o = o.mean()
            o.backward()
            assert torch.max(torch.abs(x.grad)) > 0.

    # test that F_kind='decimate' works
    x = np.random.randn(128)
    for backend in ('numpy', 'torch', 'jax'):
        if cant_import(backend):
            continue
        jtfs = TimeFrequencyScattering1D(shape=len(x), J=5, Q=4,
                                         F_kind='decimate', average_fr=True,
                                         frontend=backend)
        _ = jtfs(x)


def test_misc():
    if SKIP_ALL:
        return None if run_without_pytest else pytest.skip()
    tkt.introspection._get_pair_factor('psi', 1)
    tkt.introspection._get_pair_factor('x', 1)

    if 'tensorflow' in backends:
        B = ExtendedUnifiedBackend('tensorflow')
        x = tf.random.normal((4, 5))
        _ = B.median(x)
        _ = B.log(x)
        _ = B.numpy(x)


# run tests ##################################################################
if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        for backend in backends:
            test_normalize(backend)
            test_tensor_padded(backend)
        test_validate_filterbank()
        test_fit_smart_paths()
        test_decimate()
        test_misc()
    else:
        pytest.main([__file__, "-s"])
