# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import pytest
import os
import numpy as np
import io

from wavespin import Scattering1D
from utils import TEST_DATA_DIR, FORCED_PYTEST

# set True to execute all test functions without pytest
run_without_pytest = 0


def test_Scattering1D_numpy():
    """
    Applies scattering on a stored signal to make sure its output agrees with
    a previously calculated version.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/
    test_numpy_scattering1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """

    with open(os.path.join(TEST_DATA_DIR, 'test_scattering_1d.npz'), 'rb'
              ) as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)

    x = data['x']
    Sx0 = data['Sx']
    J = data['J']
    Q = data['Q']
    N = x.shape[-1]

    sc = Scattering1D(N, J, Q, frontend='numpy', smart_paths='primitive')
    Sx = sc(x)

    assert np.allclose(Sx, Sx0)

    # for coverage
    sc.out_type = 'list'
    _ = sc(x)


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_Scattering1D_numpy()
    else:
        pytest.main([__file__, "-s"])
