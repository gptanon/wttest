# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Miscellaneous tests that don't fit other categories."""
import pytest
import warnings
import numpy as np

from wavespin import Scattering1D, TimeFrequencyScattering1D
from wavespin.utils.gen_utils import backend_has_gpu
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
if not cant_import('jax'):
    backends.append('jax')
    import jax


#### Tests ###################################################################
@pytest.mark.parametrize("backend", backends)
def test_precision(backend):
    """Test `precision` works as intended.

    Additionally checks `device` for torch backend, though this isn't the main
    device test.  # TODO really not main test?
    """
    N = 96
    x = np.random.randn(N)

    for is_jtfs in (False, True):
      for precision in ('single', 'double'):
        if precision == 'double' and backend == 'jax':
            continue
        for device in ('cpu', 'cuda'):
          if device == 'cuda':
            if (backend != 'torch' or
                (backend == 'torch' and not backend_has_gpu('torch'))):
              continue

          for out_type in ('list', 'array', 'dict:list', 'dict:array'):
            if 'dict:' in out_type and not is_jtfs:
                continue
            for out_3D in (False, True):
                if out_3D and not is_jtfs:
                    continue
                test_params = dict(is_jtfs=is_jtfs, precision=precision,
                                   device=device, out_type=out_type)
                test_params_str = ", ".join(
                    f"{k}={v}" for k, v in test_params.items())

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"^.*effects and filter distortion.*$")
                    (precs_match, devices_match
                     ) = _get_precision_and_device(
                         x, is_jtfs, precision, device, out_type, out_3D,
                         backend)
                assert all(precs_match), "%s\n%s" % (
                    test_params_str, precs_match)
                assert all(devices_match), "%s\n%s" % (
                    test_params_str, devices_match)


def _get_precision_and_device(x, is_jtfs, precision, device, out_type, out_3D,
                              backend):
    expected_dtype = ('float64' if precision == 'double' else
                      'float32')
    ckw = dict(shape=len(x), precision=precision, out_type=out_type,
               frontend=backend, max_pad_factor=0)

    if not is_jtfs:
        sc = Scattering1D(**ckw)
        if device == 'cuda':
            sc.gpu()
        o = sc(x)

        if out_type == 'list':
            coeffs = [c['coef'] for c in o]
        else:
            coeffs = [o]

    else:
        jtfs = TimeFrequencyScattering1D(**ckw, average_fr=True, out_3D=out_3D,
                                         max_pad_factor_fr=0)
        if device == 'cuda':
            jtfs.gpu()
        o = jtfs(x)

        if out_type == 'list':
            if out_3D:
                coeffs = ([c['coef'] for c in o[0]] +
                          [c['coef'] for c in o[1]])
            else:
                coeffs = [c['coef'] for c in o]
        elif out_type == 'array':
            if out_3D:
                coeffs = o
            else:
                coeffs = [o]
        elif out_type == 'dict:list':
            coeffs = [c['coef'] for pair in o for c in o[pair]]
        elif out_type == 'dict:array':
            coeffs = list(o.values())

    precs_match = [expected_dtype in str(c.dtype) for c in coeffs]
    if backend == 'torch':
        devices_match = [device in str(c.device) for c in coeffs]
    else:
        devices_match = [True]
    return precs_match, devices_match


# run tests ##################################################################
if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        for backend in backends:
            test_precision(backend)
    else:
        pytest.main([__file__, "-s"])
