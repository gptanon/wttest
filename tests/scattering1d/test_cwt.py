# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Continuous Wavelet Transform-related tests."""
import pytest
import numpy as np

from wavespin import Scattering1D
from wavespin.utils.gen_utils import npy
from utils import cant_import, FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 1


def test_vs_scattering():
    """Assert agreement of modulus with that of first-order scattering,
    for all `precision`.
    """
    # also test unpadding by picking non-dyadic `N`
    N = 439
    x = np.random.randn(N)

    for precision in (None, 'single', 'double'):
        sc = Scattering1D(N, Q=8, J=6, average=False, out_type='list',
                          oversampling=99, precision=precision)
        meta = sc.meta()

        o0 = sc(x)
        o0 = np.array([c['coef'] for i, c in enumerate(o0)
                       if meta['order'][i] == 1])
        o1 = sc.cwt(x)
        o1 = np.abs(o1)

        assert o0.dtype == o1.dtype, (o0.dtype, o1.dtype, precision)
        assert o0.shape == o1.shape, (o0.shape, o1.shape, precision)
        assert np.allclose(o0, o1), precision


def test_hop_size():
    """Essentially that it does `cwt(x)[..., ::hop_size]`, also test backends."""
    N = 512
    hop_size = 8
    x = np.random.randn(N)

    for backend in ('numpy', 'torch', 'tensorflow'):
        if cant_import(backend):
            continue

        sc = Scattering1D(N, Q=8, J=6, frontend=backend)

        o0 = sc.cwt(x)[..., ::hop_size]
        o1 = sc.cwt(x, hop_size)
        o0n, o1n = npy(o0), npy(o1)

        assert o0n.shape == o1n.shape, (o0n.shape, o1n.shape, backend)
        # cause tensorflow
        atol = 1e-8 if backend != 'tensorflow' else 2e-6
        assert np.allclose(o0n, o1n, atol=atol), backend


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_vs_scattering()
        test_hop_size()
    else:
        pytest.main([__file__, "-s"])
