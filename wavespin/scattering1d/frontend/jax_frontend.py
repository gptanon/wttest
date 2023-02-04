# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
from ...frontend.jax_frontend import ScatteringJax
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase1D
from .frontend_utils import _handle_args_jtfs, _to_device


class ScatteringJax1D(ScatteringJax, ScatteringBase1D):
    def __init__(self, shape, J=None, Q=8, T=None, average=True, oversampling=0,
                 out_type='array', pad_mode='reflect', smart_paths=.01,
                 max_order=2, vectorized=True,
                 backend='jax',
                 **kwargs):
        ScatteringJax.__init__(self)
        ScatteringBase1D.__init__(
            self, shape, J, Q, T, average, oversampling, out_type, pad_mode,
            smart_paths, max_order, vectorized, backend, **kwargs)
        ScatteringBase1D._instantiate_backend(self,
                                              'wavespin.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
        ScatteringBase1D.finish_build(self)

    def gpu(self):  # no-cov
        """Converts filters from NumPy arrays to Jax arrays on GPU."""
        self.to_device('gpu')
        self.rebuild_for_reactives()
        return self

    def cpu(self):
        """Converts filters from NumPy arrays to Jax arrays on CPU."""
        self.to_device('cpu')
        self.rebuild_for_reactives()
        return self

    def to_device(self, device):
        """Converts filters from NumPy arrays to Jax arrays on the specified
        device, which should be `'cpu'`, `'gpu'`, or a valid input to
        `jax.device_put(, device=)`.

        The idea is to spare this conversion overhead at runtime.
        """
        _to_device(self, device)
        return self

    def update_filters(self, name):
        """Handle backend-specific filter operations for a specific filter set."""
        _to_device(self, self.filters_device, name)


ScatteringJax1D._document()


class TimeFrequencyScatteringJax1D(TimeFrequencyScatteringBase1D,
                                   ScatteringJax1D):
    def __init__(self, shape, J=None, Q=8, J_fr=None, Q_fr=1, T=None, F=None,
                 average=True, average_fr=False, oversampling=0, out_type='array',
                 pad_mode='reflect', smart_paths=.007, implementation=None,
                 vectorized=True, vectorized_fr=None, backend='jax',
                 **kwargs):
        (max_order_tm, subcls_out_type, smart_paths_tm, kwargs_tm, kwargs_fr
         ) = _handle_args_jtfs(out_type, kwargs)

        # First & second-order scattering object for the time variable
        ScatteringJax1D.__init__(
            self, shape, J, Q, T, average, oversampling, subcls_out_type,
            pad_mode, smart_paths=smart_paths_tm, max_order=max_order_tm,
            vectorized=vectorized, backend=backend,
            **kwargs_tm)

        # Frequential scattering object
        TimeFrequencyScatteringBase1D.__init__(
            self, J_fr, Q_fr, F, average_fr, out_type, smart_paths,
            vectorized_fr, implementation, **kwargs_fr)
        TimeFrequencyScatteringBase1D.build(self)


TimeFrequencyScatteringJax1D._document()


__all__ = ['ScatteringJax1D', 'TimeFrequencyScatteringJax1D']
