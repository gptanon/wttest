# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import torch

from ...frontend.torch_frontend import ScatteringTorch
from .base_frontend import ScatteringBase1D, TimeFrequencyScatteringBase1D
from ..scat_utils import _handle_args_jtfs


class ScatteringTorch1D(ScatteringTorch, ScatteringBase1D):
    """PyTorch frontend object.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/frontend/
    torch_frontend.py
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
        if register_filters:
            self.register_filters()

    def register_filters(self):
        """Registers NumPy array filters as `Module`'s buffers, to be loaded
        later at the correct device.
        """
        def do_register(p_f, n):
            p_f = torch.from_numpy(p_f).float()
            self.register_buffer('tensor' + str(n), p_f)

        n = 0
        # prepare for pytorch
        for k in self.phi_f.keys():
            if isinstance(k, int):
                do_register(self.phi_f[k], n)
                n += 1
        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                if isinstance(sub_k, int):
                    # iterate anyway so `n` is consistent
                    if not self.vectorized_early_U_1:
                        do_register(psi_f[sub_k], n)
                    n += 1
        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                if isinstance(sub_k, int):
                    do_register(psi_f[sub_k], n)
                    n += 1

        if self.vectorized_early_U_1:
            self.register_buffer('tensor_vectorized',
                                 torch.from_numpy(self.psi1_f_stacked).float())

    def load_filters(self):
        """This function loads filters from the module's buffer """
        buffer_dict = dict(self.named_buffers())
        n = 0

        for k in self.phi_f.keys():
            if isinstance(k, int):
                self.phi_f[k] = buffer_dict['tensor' + str(n)]
                n += 1

        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                if isinstance(sub_k, int):
                    if not self.vectorized_early_U_1:
                        psi_f[sub_k] = buffer_dict['tensor' + str(n)]
                    n += 1

        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                if isinstance(sub_k, int):
                    psi_f[sub_k] = buffer_dict['tensor' + str(n)]
                    n += 1

        if self.vectorized_early_U_1:
            self.psi1_f_stacked = buffer_dict['tensor_vectorized']


ScatteringTorch1D._document()


class TimeFrequencyScatteringTorch1D(TimeFrequencyScatteringBase1D,
                                     ScatteringTorch1D):
    def __init__(self, shape, J=None, Q=8, J_fr=None, Q_fr=2, T=None, F=None,
                 average=True, average_fr=False, oversampling=0, out_type="array",
                 pad_mode='reflect', smart_paths=.01, implementation=None,
                 backend="torch",
                 **kwargs):
        (max_order_tm, subcls_out_type, smart_paths_tm, kwargs_tm, kwargs_fr
         ) = _handle_args_jtfs(out_type, kwargs)

        # First & second-order scattering object for the time variable
        ScatteringTorch1D.__init__(
            self, shape, J, Q, T, average, oversampling, subcls_out_type,
            pad_mode, smart_paths=smart_paths_tm, max_order=max_order_tm,
            backend=backend, register_filters=False,
            **kwargs_tm)

        # Frequential scattering object
        TimeFrequencyScatteringBase1D.__init__(
            self, J_fr, Q_fr, F, average_fr, out_type, smart_paths,
            implementation, **kwargs_fr)
        TimeFrequencyScatteringBase1D.build(self)

        self.register_filters()

    def register_filters(self):
        """ This function run the filterbank function that
        will create the filters as numpy array, and then, it
        saves those arrays as module's buffers."""
        n_final = self._register_filters(self, ('phi_f', 'psi1_f', 'psi2_f'))
        # register filters from freq-scattering object (see base_frontend.py)
        self._register_filters(self.scf,
                               ('phi_f_fr', 'psi1_f_fr_up', 'psi1_f_fr_dn'),
                               n0=n_final)

    def _register_filters(self, obj, filter_names, n0=0):
        n = n0
        for name in filter_names:
            p_f = getattr(obj, name)
            if name.startswith('psi') and 'fr' not in name:
                for n_tm in range(len(p_f)):
                    for k in p_f[n_tm]:
                        if not isinstance(k, int):
                            continue
                        p_f[n_tm][k] = torch.from_numpy(p_f[n_tm][k]).float()
                        self.register_buffer(f'tensor{n}', p_f[n_tm][k])
                        n += 1
            elif name.startswith('psi') and 'fr' in name:
                for psi_id in p_f:
                    if not isinstance(psi_id, int):
                        continue
                    for n1_fr in range(len(p_f[psi_id])):
                        p_f[psi_id][n1_fr] = torch.from_numpy(
                            p_f[psi_id][n1_fr]).float()
                        self.register_buffer(f'tensor{n}', p_f[psi_id][n1_fr])
                        n += 1
            elif name == 'phi_f':
                for trim_tm in p_f:
                    if not isinstance(trim_tm, int):
                        continue
                    for k in range(len(p_f[trim_tm])):
                        p_f[trim_tm][k] = torch.from_numpy(p_f[trim_tm][k]
                                                           ).float()
                        self.register_buffer(f'tensor{n}', p_f[trim_tm][k])
                        n += 1
            elif name == 'phi_f_fr':
                for log2_F_phi_diff in p_f:
                    if not isinstance(log2_F_phi_diff, int):
                        continue
                    for pad_diff in p_f[log2_F_phi_diff]:
                        for sub in range(len(p_f[log2_F_phi_diff][pad_diff])):
                            p_f[log2_F_phi_diff][pad_diff][sub] = (
                                torch.from_numpy(
                                    p_f[log2_F_phi_diff][pad_diff][sub]).float())
                            self.register_buffer(
                                f'tensor{n}', p_f[log2_F_phi_diff][pad_diff][sub])
                            n += 1
            else:
                raise ValueError("unknown filter name: %s" % name)

        n_final = n
        return n_final

    def load_filters(self):
        """This function loads filters from the module's buffer """
        n_final = self._load_filters(self, ('phi_f', 'psi1_f', 'psi2_f'))
        # register filters from freq-scattering object (see base_frontend.py)
        self._load_filters(self.scf,
                           ('phi_f_fr', 'psi1_f_fr_up', 'psi1_f_fr_dn'),
                           n0=n_final)

    def _load_filters(self, obj, filter_names, n0=0):
        buffer_dict = dict(self.named_buffers())
        n = n0
        for name in filter_names:
            p_f = getattr(obj, name)
            if name.startswith('psi') and 'fr' not in name:
                for n_tm in range(len(p_f)):
                    for k in p_f[n_tm]:
                        if not isinstance(k, int):
                            continue
                        p_f[n_tm][k] = buffer_dict[f'tensor{n}']
                        n += 1
            elif name.startswith('psi') and 'fr' in name:
                for psi_id in p_f:
                    if not isinstance(psi_id, int):
                        continue
                    for n1_fr in range(len(p_f[psi_id])):
                        p_f[psi_id][n1_fr] = buffer_dict[f'tensor{n}']
                        n += 1
            elif name == 'phi_f':
                for trim_tm in p_f:
                    if not isinstance(trim_tm, int):
                        continue
                    for k in range(len(p_f[trim_tm])):
                        p_f[trim_tm][k] = buffer_dict[f'tensor{n}']
                        n += 1
            elif name == 'phi_f_fr':
                for log2_F_phi_diff in p_f:
                    if not isinstance(log2_F_phi_diff, int):
                        continue
                    for pad_diff in p_f[log2_F_phi_diff]:
                        for sub in range(len(p_f[log2_F_phi_diff][pad_diff])):
                            p_f[log2_F_phi_diff
                                ][pad_diff][sub] = buffer_dict[f'tensor{n}']
                            n += 1
            else:
                raise ValueError("unknown filter name: %s" % name)

        n_final = n
        return n_final


TimeFrequencyScatteringTorch1D._document()


__all__ = ['ScatteringTorch1D', 'TimeFrequencyScatteringTorch1D']
