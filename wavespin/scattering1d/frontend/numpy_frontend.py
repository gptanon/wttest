# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import warnings
from ...frontend.numpy_frontend import ScatteringNumPy
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase1D
from .frontend_utils import _handle_args_jtfs


class ScatteringNumPy1D(ScatteringNumPy, ScatteringBase1D):
    """NumPy frontend object.

    This is a modification of `kymatio/scattering1d/frontend/numpy_frontend.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def __init__(self, shape, J=None, Q=8, T=None, average=True, oversampling=0,
                 out_type='array', pad_mode='reflect', smart_paths=.01,
                 max_order=2, vectorized=True,
                 backend='numpy',
                 **kwargs):
        ScatteringNumPy.__init__(self)
        ScatteringBase1D.__init__(
            self, shape, J, Q, T, average, oversampling, out_type, pad_mode,
            smart_paths, max_order, vectorized, backend, **kwargs)
        ScatteringBase1D._instantiate_backend(self,
                                              'wavespin.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
        ScatteringBase1D.finish_build(self)

    def gpu(self):  # no-cov
        """Invalid for NumPy backend. Refer to other backends' docstrings.
        """
        raise Exception("NumPy backend doesn't support GPU execution.")

    def cpu(self):  # no-cov
        """Non-functional for NumPy backend. Refer to other backends' docstrings.
        """
        warnings.warn("NumPy backend is always on CPU, so `cpu()` does nothing.")


ScatteringNumPy1D._document()


class TimeFrequencyScatteringNumPy1D(TimeFrequencyScatteringBase1D,
                                     ScatteringNumPy1D):
    def __init__(self, shape, J=None, Q=8, J_fr=None, Q_fr=1, T=None, F=None,
                 average=True, average_fr=False, oversampling=0, out_type="array",
                 pad_mode='reflect', smart_paths=.007, implementation=None,
                 vectorized=True, vectorized_fr=None, backend="numpy",
                 **kwargs):
        (max_order_tm, subcls_out_type, smart_paths_tm, kwargs_tm, kwargs_fr
         ) = _handle_args_jtfs(out_type, kwargs)

        # First & second-order scattering object for the time variable
        ScatteringNumPy1D.__init__(
            self, shape, J, Q, T, average, oversampling, subcls_out_type,
            pad_mode, smart_paths=smart_paths_tm, max_order=max_order_tm,
            vectorized=vectorized, backend=backend,
            **kwargs_tm)

        # Frequential scattering object
        TimeFrequencyScatteringBase1D.__init__(
            self, J_fr, Q_fr, F, average_fr, out_type, smart_paths,
            vectorized_fr, implementation, **kwargs_fr)
        TimeFrequencyScatteringBase1D.build(self)


TimeFrequencyScatteringNumPy1D._document()


__all__ = ['ScatteringNumPy1D', 'TimeFrequencyScatteringNumPy1D']
