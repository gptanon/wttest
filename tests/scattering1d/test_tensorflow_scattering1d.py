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

from utils import cant_import, TEST_DATA_DIR, FORCED_PYTEST

# skip this test if no TF installed
got_tf = bool(not cant_import('tensorflow'))

# set True to execute all test functions without pytest
run_without_pytest = 0


if got_tf:
    from wavespin.tensorflow import Scattering1D


def test_Scattering1D_tensorflow():
    """
    Applies scattering on a stored signal to make sure its output agrees with
    a previously calculated version.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/tests/
    test_numpy_scattering1d.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    if not got_tf:
        return None if run_without_pytest else pytest.skip()
    import tensorflow as tf

    with open(os.path.join(TEST_DATA_DIR, 'test_scattering_1d.npz'), 'rb'
              ) as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)

    x = data['x']
    Sx0 = data['Sx']
    J = data['J']
    Q = data['Q']
    N = x.shape[-1]

    sc = Scattering1D(J, N, Q)

    Sx = sc(x)
    adiff = tf.math.abs(Sx - Sx0)
    # TF is quite imprecise
    assert np.allclose(Sx, Sx0, atol=8e-6, rtol=1e-7), (
        tf.reduce_mean(adiff).numpy(), tf.reduce_max(adiff).numpy())

    # for coverage
    sc.out_type = 'list'
    _ = sc(x)


if __name__ == '__main__':
    if run_without_pytest and not FORCED_PYTEST:
        test_Scattering1D_tensorflow()
    else:
        pytest.main([__file__, "-s"])
