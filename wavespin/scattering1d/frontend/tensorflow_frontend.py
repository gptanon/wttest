# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
from ...frontend.tensorflow_frontend import ScatteringTensorFlow
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase1D
from .frontend_utils import _handle_args_jtfs, _to_device


class ScatteringTensorFlow1D(ScatteringTensorFlow, ScatteringBase1D):
    """TensorFlow frontend object.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/frontend/
    tensorflow_frontend.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def __init__(self, shape, J=None, Q=8, T=None, average=True, oversampling=0,
                 out_type='array', pad_mode='reflect', smart_paths=.01,
                 max_order=2, vectorized=True,
                 backend='tensorflow', name='Scattering1D',
                 **kwargs):
        ScatteringTensorFlow.__init__(self, name=name)
        ScatteringBase1D.__init__(
            self, shape, J, Q, T, average, oversampling, out_type, pad_mode,
            smart_paths, max_order, vectorized, backend, **kwargs)
        ScatteringBase1D._instantiate_backend(self,
                                              'wavespin.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
        ScatteringBase1D.finish_build(self)

    def gpu(self):
        """Converts filters from NumPy arrays to TensorFlow tensors on GPU."""
        self.to_device('gpu')
        return self

    def cpu(self):
        """Converts filters from NumPy arrays to TensorFlow tensors on CPU."""
        self.to_device('cpu')
        return self

    def to_device(self, device):
        """Converts filters from NumPy arrays to TensorFlow tensors on the
        specified device, which should be `'cpu'`, `'gpu'`, or a valid input
        to `tf.device(device_name=)`.

        The idea is to spare this conversion overhead at runtime.
        """
        _to_device(self, device)


ScatteringTensorFlow1D._document()


class TimeFrequencyScatteringTensorFlow1D(TimeFrequencyScatteringBase1D,
                                          ScatteringTensorFlow1D):
    def __init__(self, shape, J=None, Q=8, J_fr=None, Q_fr=2, T=None, F=None,
                 average=True, average_fr=False, oversampling=0, out_type="array",
                 pad_mode='reflect', smart_paths=.007, implementation=None,
                 backend="tensorflow", name='TimeFrequencyScattering1D',
                 **kwargs):
        (max_order_tm, subcls_out_type, smart_paths_tm, kwargs_tm, kwargs_fr
         ) = _handle_args_jtfs(out_type, kwargs)

        # First & second-order scattering object for the time variable
        ScatteringTensorFlow1D.__init__(
            self, shape, J, Q, T, average, oversampling, subcls_out_type,
            pad_mode, smart_paths=smart_paths_tm, max_order=max_order_tm,
            vectorized=False, backend=backend,
            **kwargs_tm)

        # Frequential scattering object
        TimeFrequencyScatteringBase1D.__init__(
            self, J_fr, Q_fr, F, average_fr, out_type, smart_paths,
            implementation, **kwargs_fr)
        TimeFrequencyScatteringBase1D.build(self)


TimeFrequencyScatteringTensorFlow1D._document()


__all__ = ['ScatteringTensorFlow1D', 'TimeFrequencyScatteringTensorFlow1D']
