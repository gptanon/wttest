# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
from ...frontend.torch_frontend import ScatteringTorch
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase1D
from .frontend_utils import _handle_args_jtfs, _to_device, _tensor_name


class ScatteringTorch1D(ScatteringTorch, ScatteringBase1D):
    """PyTorch frontend object.

    This is a modification of `kymatio/scattering1d/frontend/torch_frontend.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def __init__(self, shape, J=None, Q=8, T=None, average=True, oversampling=0,
                 out_type='array', pad_mode='reflect', smart_paths=.01,
                 max_order=2, vectorized=True,
                 backend='torch', register_filters=True,
                 **kwargs):
        ScatteringTorch.__init__(self)
        ScatteringBase1D.__init__(
            self, shape, J, Q, T, average, oversampling, out_type, pad_mode,
            smart_paths, max_order, vectorized, backend, **kwargs)
        ScatteringBase1D._instantiate_backend(self,
                                              'wavespin.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
        ScatteringBase1D.finish_build(self)
        if register_filters:
            self.register_filters()

    def register_filters(self, names=None):
        """Registers filters as `nn.Module`'s `named_buffers()`. They're later
        loaded by `load_filters()` at runtime.

        The idea is, when `cuda()` or `cpu()` is called, the filters stored in
        `named_buffers` are moved to GPU or CPU, respectively, and then fetched
        for runtime computation. It's also to expose the filters as usual
        (but non-trainable) model parameters.
        """
        _to_device(self, names)

    def gpu(self):  # no-cov
        """Converts filters from NumPy arrays to PyTorch tensors on GPU.
        """
        return self._handle_device('gpu')

    def cpu(self):  # no-cov
        """Converts filters from NumPy arrays to PyTorch tensors on CPU.
        """
        return self._handle_device('cpu')

    def to_device(self, device):
        """Converts filters from NumPy arrays to PyTorch tensors on the
        specified device, which should be `'cpu'`, `'gpu'`, or a valid input
        to `torch.Tensor.to(device=)`.

        The idea is to spare this conversion overhead at runtime.
        """
        return self._handle_device(device)

    def to(self, *args, **kwargs):
        raise Exception("`to()` won't always work as intended, e.g. changing "
                        "`dtype` won't resample filters. Use `to_device()`.")

    def _handle_device(self, device):
        if isinstance(device, str) and device in ('gpu', 'cpu'):
            if device == 'gpu':
                self.cuda()
            elif device == 'cpu':
                ScatteringTorch.to(self, device='cpu')
        else:
            ScatteringTorch.to(self, device=device)
        self.load_filters()
        self.pack_runtime_filters()
        return self

    def load_filters(self, names=None):
        """Loads filters from `torch.nn.Module`'s buffer."""
        if names is None:
            names = ('phi_f', 'psi1_f', 'psi2_f', 'psi1_f_stacked')
        elif not isinstance(names, (list, tuple)):
            names = [names]
        buffer_dict = dict(self.named_buffers())

        if 'phi_f' in names:
            for k in self.phi_f:
                if isinstance(k, int):
                    self.phi_f[k] = buffer_dict[_tensor_name('phi_f', k)]

        if 'psi1_f' in names and not self.vectorized_early_U_1:
            for n1, psi_f in enumerate(self.psi1_f):
                for k in psi_f:
                    if isinstance(k, int):
                        psi_f[k] = buffer_dict[_tensor_name('psi1_f', n1, k)]

        if 'psi2_f' in names:
            for n2, psi_f in enumerate(self.psi2_f):
                for k in psi_f:
                    if isinstance(k, int):
                        psi_f[k] = buffer_dict[_tensor_name('psi2_f', n2, k)]

        if 'psi1_f_stacked' in names and self.vectorized_early_U_1:
            self.psi1_f_stacked = buffer_dict[_tensor_name('psi1_f_stacked')]

    def update_filters(self, name):
        """Handle backend-specific filter operations for a specific filter set."""
        # first we unregister filters
        for a in dir(self):
            if a.startswith('t_' + name):
                delattr(self, a)

        # now re-register
        _to_device(self, self.filters_device, [name])
        # filters are in buffer but haven't moved to device yet
        ScatteringTorch.to(self, device=self.filters_device)
        # finally, fetch from buffer
        self.load_filters([name])


ScatteringTorch1D._document()


class TimeFrequencyScatteringTorch1D(TimeFrequencyScatteringBase1D,
                                     ScatteringTorch1D):
    def __init__(self, shape, J=None, Q=8, J_fr=None, Q_fr=1, T=None, F=None,
                 average=True, average_fr=False, oversampling=0, out_type="array",
                 pad_mode='reflect', smart_paths=.007, implementation=None,
                 vectorized=True, vectorized_fr=None, backend="torch",
                 **kwargs):
        (max_order_tm, subcls_out_type, smart_paths_tm, kwargs_tm, kwargs_fr
         ) = _handle_args_jtfs(out_type, kwargs)

        # First & second-order scattering object for the time variable
        ScatteringTorch1D.__init__(
            self, shape, J, Q, T, average, oversampling, subcls_out_type,
            pad_mode, smart_paths=smart_paths_tm, max_order=max_order_tm,
            vectorized=vectorized, backend=backend, register_filters=False,
            **kwargs_tm)

        # Frequential scattering object
        TimeFrequencyScatteringBase1D.__init__(
            self, J_fr, Q_fr, F, average_fr, out_type, smart_paths,
            vectorized_fr, implementation, **kwargs_fr)
        TimeFrequencyScatteringBase1D.build(self)

        self.register_filters()

    def load_filters(self, names=None):
        """Loads filters from the module's buffer. Also see `register_filters`."""
        names_tm = ('phi_f', 'psi1_f', 'psi2_f', 'psi1_f_stacked')
        names_fr = ('phi_f_fr', 'psi1_f_fr_up', 'psi1_f_fr_dn',
                    'psi1_f_fr_stacked_dict')
        if names is None:
            self._load_filters(self, names_tm)
            # register filters from freq-scattering object (see base_frontend.py)
            self._load_filters(self.scf, names_fr)
        else:
            if not isinstance(names, (list, tuple)):
                names = [names]
            for name in names:
                obj = (self if name in names_tm else
                       self.scf)
                self._load_filters(obj, name)

    def _load_filters(self, obj, filter_names):
        if not isinstance(filter_names, (list, tuple)):
            filter_names = [filter_names]
        buffer_dict = dict(self.named_buffers())

        for name in filter_names:
            p_f = getattr(obj, name)

            if name == 'psi1_f_stacked':
                if self.vectorized_early_U_1:
                    setattr(self, name, buffer_dict[_tensor_name(name)])

            elif name == 'psi1_f_fr_stacked_dict':
                if self.vectorized_fr:
                    for psi_id in p_f:
                        if self.vectorized_early_fr:
                            p_f[psi_id] = buffer_dict[_tensor_name(name, psi_id)]
                        else:
                            for n1_fr_subsample in p_f[psi_id]:
                                _nm = _tensor_name(name, psi_id, n1_fr_subsample)
                                p_f[psi_id][n1_fr_subsample] = buffer_dict[_nm]

            elif name in ('psi1_f', 'psi2_f'):
                if (name == 'psi2_f' or
                    (name == 'psi1_f' and not self.vectorized_early_U_1)):
                    for n_tm in range(len(p_f)):
                        for k in p_f[n_tm]:
                            if not isinstance(k, int):
                                continue
                            _nm = _tensor_name(name, n_tm, k)
                            p_f[n_tm][k] = buffer_dict[_nm]

            elif name.startswith('psi') and 'fr' in name:
                if not self.vectorized_fr:
                    for psi_id in p_f:
                        if not isinstance(psi_id, int):
                            continue
                        for n1_fr in range(len(p_f[psi_id])):
                            _nm = _tensor_name(name, psi_id, n1_fr)
                            p_f[psi_id][n1_fr] = buffer_dict[_nm]

            elif name == 'phi_f':
                for trim_tm in p_f:
                    if not isinstance(trim_tm, int):
                        continue
                    for k in range(len(p_f[trim_tm])):
                        _nm = _tensor_name(name, trim_tm, k)
                        p_f[trim_tm][k] = buffer_dict[_nm]

            elif name == 'phi_f_fr':
                for log2_F_phi_diff in p_f:
                    if not isinstance(log2_F_phi_diff, int):
                        continue
                    for pad_diff in p_f[log2_F_phi_diff]:
                        for sub in range(len(p_f[log2_F_phi_diff][pad_diff])):
                            _nm = _tensor_name(name, log2_F_phi_diff, pad_diff,
                                               sub)
                            p_f[log2_F_phi_diff][pad_diff][sub] = buffer_dict[_nm]

            else:  # no-cov
                raise ValueError("unknown filter name: %s" % name)


TimeFrequencyScatteringTorch1D._document()


__all__ = ['ScatteringTorch1D', 'TimeFrequencyScatteringTorch1D']
