# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Miscellaneous tests that don't fit other categories."""
import pytest
import numpy as np

from wavespin import Scattering1D, TimeFrequencyScattering1D
from utils import cant_import, FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 1

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
def test_precision(backend):
    """Test `precision` works as intended.

    Additionally checks `device` for torch backend, though this isn't the main
    device test.
    """
    N = 128
    x = np.random.randn(N)

    precision_to_dtype = {'single': 'float32', 'double': 'float64'}

    for precision in ('single', 'double'):
        for device in ('cpu', 'cuda'):
            if device == 'cuda':
                if (backend != 'torch' or
                    (backend == 'torch' and not torch.cuda.is_available())):
                    continue
            ckw = dict(shape=N, frontend=backend, precision=precision,
                       out_type='array')

            sc = Scattering1D(**ckw)
            jtfs = TimeFrequencyScattering1D(**ckw, average_fr=True)
            if device == 'cuda':
                sc.gpu()
                jtfs.gpu()

            o_sc = sc(x)
            o_jtfs = jtfs(x)

            def assert_in(a, b):
                assert a in b, (a, b)

            # check dtype
            expected_dtype = precision_to_dtype[precision]
            assert_in(expected_dtype, str(o_sc.dtype))
            assert_in(expected_dtype, str(o_jtfs.dtype))

            # check device
            if backend == 'torch':
                assert_in(device, str(o_sc.device))
                assert_in(device, str(o_jtfs.device))


# run tests ##################################################################
if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        for backend in backends:
            test_precision(backend)
    else:
        pytest.main([__file__, "-s"])
