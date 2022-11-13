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

from wavespin.toolkit import normalize, tensor_padded, validate_filterbank
from wavespin.utils.gen_utils import ExtendedUnifiedBackend
from wavespin import toolkit as tkt
from wavespin import Scattering1D, TimeFrequencyScattering1D
from utils import tempdir, cant_import, SKIPS, FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 1
# set True to disable matplotlib plots
# (done automatically for CI via `conftest.py`, but `False` here takes precedence)
no_plots = 1
# set True to skip this file entirely
skip_all = SKIPS['toolkit']

# set backends based on what's available
backends = ['numpy']
if not cant_import('torch'):
    backends.append('torch')
    import torch
if not cant_import('tensorflow'):
    backends.append('tensorflow')
    import tensorflow as tf


#### Tests ###################################################################

@pytest.mark.parametrize("backend", backends)
def test_normalize(backend):
    """Ensure error thrown upon invalid input, but otherwise method
    doesn't error.
    """
    if skip_all:
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
    if skip_all:
        return None if run_without_pytest else pytest.skip()

    # helper methods #########################################################
    if backend == 'numpy':
        tensor_fn = np.array
    elif backend == 'torch':
        tensor_fn = torch.tensor
    elif backend == 'tensorflow':
        tensor_fn = tf.constant

    def cond_fn(a, b):
        if backend in ('torch', 'tensorflow'):
            a, b = a.numpy(), b.numpy()
        return np.allclose(a, b)

    def tensor(x):
        if len(x[0]) != len(x[1]):  # ragged case
            if backend == 'numpy':
                return x
            return [[tensor_fn(dim1) for dim1 in dim0] for dim0 in x]
        return tensor_fn(x)

    def assert_fn(target, out):
        assert cond_fn(target, out), "{}\n{}".format(backend, out)

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
    if skip_all:
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


def test_misc():
    if skip_all:
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
        test_misc()
    else:
        pytest.main([__file__, "-s"])
