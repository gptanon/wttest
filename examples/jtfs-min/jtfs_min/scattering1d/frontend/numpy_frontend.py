# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
from ...frontend.numpy_frontend import ScatteringNumPy
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase1D


class ScatteringNumPy1D(ScatteringNumPy, ScatteringBase1D):
    """NumPy frontend object.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/frontend/
    numpy_frontend.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def __init__(self, shape, J=None, Q=8, T=None, average=True, oversampling=0,
                 backend='numpy'):
        ScatteringNumPy.__init__(self)
        ScatteringBase1D.__init__(
            self, shape, J, Q, T, average, oversampling, backend)
        ScatteringBase1D._instantiate_backend(self,
                                              'wavespin.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)


class TimeFrequencyScatteringNumPy1D(TimeFrequencyScatteringBase1D,
                                     ScatteringNumPy1D):
    def __init__(self, shape, J=None, Q=8, J_fr=None, Q_fr=2, T=None, F=None,
                 average=True, oversampling=0, oversampling_fr=0,
                 backend="numpy"):

        # First & second-order scattering object for the time variable -------
        ScatteringNumPy1D.__init__(self, shape, J, Q, T, average, oversampling,
                                   backend=backend)

        # Frequential scattering object --------------------------------------
        TimeFrequencyScatteringBase1D.__init__(self, J_fr, Q_fr, F)
        TimeFrequencyScatteringBase1D.build(self)


__all__ = ['ScatteringNumPy1D', 'TimeFrequencyScatteringNumPy1D']
