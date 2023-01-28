# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""`vectorized`- and `oversampling`-related tests."""
import os
import pytest
import numpy as np
import warnings
from pathlib import Path
from copy import deepcopy

from wavespin import Scattering1D, TimeFrequencyScattering1D
from wavespin.utils.gen_utils import npy, backend_has_gpu
from wavespin import toolkit as tkt
from wavespin.toolkit import echirp, energy
from wavespin.visuals import (coeff_distance_jtfs, compare_distances_jtfs,
                              energy_profile_jtfs, plot, plotscat)
from utils import (cant_import, IgnoreWarnings,
                   SKIPS, TEST_DATA_DIR, FORCED_PYTEST)

# backend to use for all tests except backend-specific tests
default_backend = ('numpy', 'torch', 'tensorflow', 'jax')[0]
# precision to use for all but precision-sensitive tests
default_precision = 'single'
# set True to execute all test functions without pytest
run_without_pytest = 1


def test_scattering():
    """Test that for any given `oversampling`, outputs of `vectorized=True`
    and `vectorized=False` agree, for `Scattering1D`.
    """
    N = 497
    x = np.random.randn(N)
    base_kw = dict(shape=N, Q=8)

    backends = ((default_backend, 'torch') if default_backend != 'torch' else
                ('numpy', 'torch'))

    for backend in backends:
        ckw = dict(**base_kw, frontend=backend)
        sc0 = Scattering1D(**ckw, vectorized=True)
        sc1 = Scattering1D(**ckw, vectorized=False)

        for oversampling in (0, 1, 5):
            for sc in (sc0, sc1):
                sc.update(oversampling=oversampling)

            o0 = npy(sc0(x))
            o1 = npy(sc1(x))
            assert np.allclose(o0, o1)

            if backend == 'torch':
                sc0.gpu()
                sc1.gpu()
                o0 = npy(sc0(x))
                o1 = npy(sc1(x))
                assert np.allclose(o0, o1)


def test_jtfs():
    """`test_scattering()` but for JTFS, including `vectorized_fr` and
    `oversampling_fr`.
    """
    N = 497
    x = np.random.randn(N)
    base_kw = dict(shape=N, Q=8, max_pad_factor_fr=1, out_type='array')
    vec_kw = [dict(vectorized=True,  vectorized_fr=True),
              dict(vectorized=False, vectorized_fr=False),
              dict(vectorized=True,  vectorized_fr=2)]
    oversamplings = (0, 1, 5)
    oversampling_frs = (0, 1, 3)

    backends = ((default_backend, 'torch') if default_backend != 'torch' else
                ('numpy', 'torch'))

    for backend in backends[1:]:
      for out_3D in (False, True):
        for average_fr in (False, True):
            if out_3D and not average_fr:  # invalid config
                continue
            cfg_kw = dict(frontend=backend, out_3D=out_3D,
                          average_fr=average_fr)
            ckw = dict(**base_kw, **cfg_kw)

            try:
                jtfs0 = TimeFrequencyScattering1D(**ckw, **vec_kw[0])
                jtfs1 = TimeFrequencyScattering1D(**ckw, **vec_kw[1])
                jtfs2 = TimeFrequencyScattering1D(**ckw, **vec_kw[2])

                for oversampling in oversamplings:
                  for oversampling_fr in oversampling_frs:
                      for jtfs in (jtfs0, jtfs1, jtfs2):
                          jtfs.update(oversampling=oversampling,
                                      oversampling_fr=oversampling_fr)

                      o0 = jtfs0(x)
                      o1 = jtfs1(x)
                      o2 = jtfs2(x)

                      _jtfs_run_asserts(jtfs0, jtfs1, jtfs2, o0, o1, o2, ckw)

                      if backend == 'torch':
                          jtfs0.gpu()
                          jtfs1.gpu()
                          jtfs2.gpu()

                          o0 = jtfs0(x)
                          o1 = jtfs1(x)
                          o2 = jtfs2(x)

                          _jtfs_run_asserts(jtfs0, jtfs1, jtfs2, o0, o1, o2, ckw)

            except Exception as e:
                try:
                    cfg_kw.update(oversampling=oversampling,
                                  oversampling_fr=oversampling_fr)
                except:
                    # means we failed at __init__
                    pass
                kw_text = '\n'.join(f"  {k}={v}" for k, v in cfg_kw.items())
                print("Failed with\n" + kw_text)
                raise e


def _jtfs_run_asserts(jtfs0, jtfs1, jtfs2, o0, o1, o2, ckw):
    if ckw['out_3D']:
        o0_0, o0_1 = [npy(o) for o in o0]
        o1_0, o1_1 = [npy(o) for o in o1]
        o2_0, o2_1 = [npy(o) for o in o1]
    else:
        o0, o1, o2 = [npy(o) for o in (o0, o1, o2)]

    if ckw['out_3D']:
        assert np.allclose(o0_0, o1_0)
        assert np.allclose(o0_0, o2_0)
        assert np.allclose(o0_1, o1_1)
        assert np.allclose(o0_1, o2_1)
    else:
        assert np.allclose(o0, o1)
        assert np.allclose(o0, o2)


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        # test_scattering()
        test_jtfs()
    else:
        pytest.main([__file__, "-s"])
