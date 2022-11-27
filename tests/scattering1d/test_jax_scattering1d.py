# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import os
import io
import pytest
import numpy as np

from wavespin import Scattering1D
from utils import cant_import, TEST_DATA_DIR, FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 1

# skip this test if no jax installed
got_backend = bool(not cant_import('jax'))


def test_Scattering1D_jax():
    """
    Applies scattering on a stored signal to make sure its output agrees with
    a previously calculated version.
    """
    if not got_backend:
        return None if run_without_pytest else pytest.skip()

    with open(os.path.join(TEST_DATA_DIR, 'test_scattering_1d.npz'), 'rb'
              ) as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)

    x = data['x']
    Sx0 = data['Sx']
    J = data['J']
    Q = data['Q']
    N = x.shape[-1]
    J, Q, N = int(J), int(Q), int(N)

    sc = Scattering1D(shape=N, J=J, Q=Q, frontend='jax', smart_paths='primitive')
    Sx = sc(x)

    assert np.allclose(Sx, Sx0)

    # for coverage
    sc.out_type = 'list'
    _ = sc(x)


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_Scattering1D_jax()
    else:
        pytest.main([__file__, "-s"])
