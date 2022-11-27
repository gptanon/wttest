# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import warnings
import numpy as np
from copy import deepcopy
from ..refining import smart_paths_exclude, primitive_paths_exclude


# arg handling ###############################################################
def _check_runtime_args_common(x):
    if len(x.shape) < 1:
        raise ValueError("x should be at least 1D. Got %s" % str(x))


def _check_runtime_args_scat1d(out_type, average):
    if out_type == 'array' and not average:  # no-cov
        raise ValueError("out_type=='array' and `not average` are mutually "
                         "incompatible. Please set out_type='list'.")

    if out_type not in ('array', 'list'):  # no-cov
        raise RuntimeError("`out_type` must be one of: 'array', 'list'. "
                           "Got %s" % out_type)


def _check_runtime_args_jtfs(average, average_fr, out_type, out_3D):
    if 'array' in out_type and not average:  # no-cov
        raise ValueError("Options `average=False` and `'array' in out_type` "
                         "are mutually incompatible. "
                         "Please set out_type='list' or 'dict:list'")

    if out_3D and not average_fr:  # no-cov
        raise ValueError("`out_3D=True` requires `average_fr=True`.")

    supported = ('array', 'list', 'dict:array', 'dict:list')
    if out_type not in supported:  # no-cov
        raise RuntimeError("`out_type` must be one of: {} (got {})".format(
            ', '.join(supported), out_type))


def _handle_input_and_backend(self, x):
    """
        - standardizes input shape, `(*batch_shape, time)`
        - ensures `x` is a tensor/array of correct dtype and device
        - fetches `batch_shape`
        - executes backend-specific preparation (e.g. `load_filters()` for torch)
    """
    if self.frontend_name == 'torch':
        import torch
        backend_obj = torch
    elif self.frontend_name == 'tensorflow':
        import tensorflow as tf
        backend_obj = tf
    elif self.frontend_name == 'jax':
        import jax
        backend_obj = jax.numpy
    else:
        backend_obj = np

    # ensure shape's as expected
    batch_shape = x.shape[:-1]
    signal_shape = x.shape[-1:]
    if self.frontend_name != 'tensorflow':
        x = x.reshape((-1, 1) + signal_shape)
    else:
        x = tf.reshape(x, tf.concat(((-1, 1), signal_shape), 0))

    # handle backends
    is_jtfs = bool(hasattr(self, 'scf'))
    numpy_input = bool(isinstance(x, np.ndarray))
    # refernce filter
    get_p_ref = lambda: self.phi_f[0] if not is_jtfs else self.phi_f[0][0]

    if self.frontend_name == 'torch':
        self.load_filters()

        # convert input to tensor if it isn't already
        p_ref = get_p_ref()
        device = p_ref.device.type
        if numpy_input:
            x = torch.from_numpy(x).to(device=device)
        if x.device.type != device:
            x = x.to(device)

    elif self.frontend_name in ('tensorflow', 'jax'):
        if not hasattr(self, 'on_device'):
            self.cpu()
        if numpy_input:
            p_ref = get_p_ref()
            if self.frontend_name == 'tensorflow':
                with tf.device(p_ref.device):
                    x = tf.convert_to_tensor(x)
            elif self.frontend_name == 'jax':
                x = jax.device_put(x, device=p_ref.device())

    # handle precision
    expected_dtype = {'single': 'float32', 'double': 'float64'
                      }[self.precision]
    x = self.backend.ensure_dtype(x, expected_dtype)

    return x, batch_shape, backend_obj


def _restore_batch_shape(Scx, batch_shape, frontend_name, out_type, backend_obj,
                         out_3D=None, out_structure=None, is_jtfs=False):
    # handle JTFS ############################################################
    if is_jtfs:
        if len(batch_shape) == 1:
            return Scx  # good to go
        elif out_structure is not None:
            if isinstance(Scx, tuple):
                Scx = list(Scx)
                for i in range(len(Scx)):
                    Scx[i] = backend_obj.reshape(
                        Scx[i], (*batch_shape, *Scx[i].shape[1:]))
                Scx = tuple(Scx)
            else:
                Scx = backend_obj.reshape(
                    Scx, (*batch_shape, *Scx.shape[1:]))
            return Scx
        else:
            args = (batch_shape, frontend_name, out_type.lstrip('dict:'),
                    backend_obj, out_3D, out_structure, is_jtfs)
            if out_type.startswith('dict:'):
                for pair in Scx:
                    Scx[pair] = _restore_batch_shape(Scx[pair], *args)
                return Scx
            elif out_3D and isinstance(Scx, tuple):
                Scx = list(Scx)
                for i in range(2):
                    Scx[i] = _restore_batch_shape(Scx[i], *args)
                Scx = tuple(Scx)
                return Scx

    # reshape ################################################################
    if frontend_name != 'tensorflow':
        if out_type.endswith('array'):
            scattering_shape = (Scx.shape[1:] if is_jtfs else
                                Scx.shape[-2:])
            new_shape = batch_shape + scattering_shape
            Scx = Scx.reshape(new_shape)
        elif out_type.endswith('list'):
            for c in Scx:
                scattering_shape = (c['coef'].shape[1:] if is_jtfs else
                                    c['coef'].shape[-1:])
                new_shape = batch_shape + scattering_shape
                c['coef'] = c['coef'].reshape(new_shape)
    else:
        tf = backend_obj
        if out_type.endswith('array'):
            scattering_shape = tuple(Scx.shape[1:] if is_jtfs else
                                     Scx.shape[-2:])
            new_shape = tf.cast(tf.concat((batch_shape, scattering_shape), 0),
                                tf.int32)
            Scx = tf.reshape(Scx, new_shape)
        elif out_type.endswith('list'):
            for c in Scx:
                scattering_shape = tuple(c['coef'].shape[1:] if is_jtfs else
                                         c['coef'].shape[-1:])
                new_shape = tf.cast(tf.concat((batch_shape, scattering_shape), 0),
                                    tf.int32)
                c['coef'] = tf.reshape(c['coef'], new_shape)

    return Scx


def _handle_args_jtfs(out_type, kwargs):
    from .base_frontend import ScatteringBase1D

    # subclass's `out_type`
    subcls_out_type = out_type.lstrip('dict:')
    # need for joint wavelets
    max_order_tm = 2
    # time scattering object shouldn't process any smart paths
    smart_paths_tm = 0

    # split tm & JTFS
    kwargs = deepcopy(kwargs)  # don't affect original
    for_jtfs = ('paths_exclude',)  # redirected kwargs
    kwargs_tm, kwargs_fr = {}, {}
    for name in ScatteringBase1D.SUPPORTED_KWARGS:
        if name in kwargs and name not in for_jtfs:
            kwargs_tm[name] = kwargs.pop(name)
    kwargs_fr = kwargs

    return max_order_tm, subcls_out_type, smart_paths_tm, kwargs_tm, kwargs_fr


def _ensure_positive_integer(self, names):
    def is_int(x):
        return isinstance(x, int) or np.issubdtype(x.dtype, np.integer)

    for name in names:
        value = getattr(self, name)
        if value is None:
            continue
        elif isinstance(value, (tuple, list)):  # no-cov
            for v in value:
                if v is None:
                    continue
                elif not is_int(v):
                    raise TypeError(f"`{name}` must consist of integers, found "
                                    f"{type(v)}")
                elif v < 1:
                    raise ValueError(f"`{name}` must consist of positive values, "
                                     f"found {v}")
        elif isinstance(value, str):
            if value != 'global':  # no-cov
                if name in ('T', 'F'):
                    raise ValueError(f"`{name}`, if str, must be `'global'`, "
                                     f"got {value}")
                else:
                    raise TypeError(f"`{name}` must be int, got str")
        else:  # no-cov
            if not is_int(value):
                raise TypeError(f"`{name}` must be int, got {type(value)}")
            elif value < 1:  # no-cov
                raise ValueError(f"`{name}` must be positive, got {value}")

# device handling ############################################################
def _to_device(self, device=None):
    # make `to_device` to apply to each filter ###############################
    if self.frontend_name == 'torch':
        # unlike with other frontends, here we don't move filters, rather we
        # register them under `torch.nn.Module`, which then moves them to the
        # appropriate device at runtime via `load_device()` after `self.cuda()`
        # or `self.cpu()` has been called. See `register_filters()` in
        # `wavespin.scattering1d.frontend.torch_frontend.ScatteringTorch1D`.
        import torch

        if (isinstance(device, str) and device.lower() == 'gpu' and
                not torch.cuda.is_available()):
            raise Exception("PyTorch could not find a GPU.")

        class ToDevice():
            def __init__(self, obj):
                self.obj = obj
                self.n = 0

            def __call__(self, p_f):
                p_f = torch.from_numpy(p_f)
                self.obj.register_buffer(f'tensor{self.n}', p_f)
                self.n += 1
                return p_f

        to_device = ToDevice(obj=self)

    if self.frontend_name == 'tensorflow':
        import tensorflow as tf

        if isinstance(device, str) and device.lower() in ('cpu', 'gpu'):
            devices = tf.config.list_physical_devices(device.upper())
            if devices == []:  # no-cov
                raise Exception("TensorFlow could not find a %s" % device.upper())
            device = devices[0].name.replace('physical_device:', '')

        def to_device(x):
            with tf.device(device):
                x = tf.identity(x)
            return x

    elif self.frontend_name == 'jax':
        import jax

        if isinstance(device, str) and device.lower() in ('cpu', 'gpu'):
            devices = jax.devices(device)
            if devices == []:  # no-cov
                raise Exception("Jax could not find a %s" % device.upper())
            device = devices[0]

        def to_device(x):
            return jax.device_put(x, device=device)

    # move filters ###########################################################
    is_jtfs = bool(hasattr(self, 'scf'))

    if is_jtfs:
        _move_filters_to_device_jtfs(
            self, to_device, ('phi_f', 'psi1_f', 'psi2_f'))
        _move_filters_to_device_jtfs(
            self.scf, to_device, ('phi_f_fr', 'psi1_f_fr_up', 'psi1_f_fr_dn'))
        self.on_device = str(self.phi_f[0][0].device)
    else:
        _move_filters_to_device(self, to_device)
        self.on_device = str(self.phi_f[0].device)


def _move_filters_to_device(self, to_device):
    for k in self.phi_f.keys():
        if isinstance(k, int):
            self.phi_f[k] = to_device(self.phi_f[k])

    for psi_f in self.psi1_f:
        for k in psi_f.keys():
            if isinstance(k, int):
                if not self.vectorized_early_U_1:
                    psi_f[k] = to_device(psi_f[k])

    for psi_f in self.psi2_f:
        for k in psi_f.keys():
            if isinstance(k, int):
                psi_f[k] = to_device(psi_f[k])

    if self.vectorized_early_U_1:
        self.psi1_f_stacked = to_device(self.psi1_f_stacked)


def _move_filters_to_device_jtfs(obj, to_device, filter_names):
    for name in filter_names:
        p_f = getattr(obj, name)
        if name.startswith('psi') and 'fr' not in name:
            for n_tm in range(len(p_f)):
                for k in p_f[n_tm]:
                    if not isinstance(k, int):
                        continue
                    p_f[n_tm][k] = to_device(p_f[n_tm][k])
        elif name.startswith('psi') and 'fr' in name:
            for psi_id in p_f:
                if not isinstance(psi_id, int):
                    continue
                for n1_fr in range(len(p_f[psi_id])):
                    p_f[psi_id][n1_fr] = to_device(p_f[psi_id][n1_fr])
        elif name == 'phi_f':
            for trim_tm in p_f:
                if not isinstance(trim_tm, int):
                    continue
                for k in range(len(p_f[trim_tm])):
                    p_f[trim_tm][k] = to_device(p_f[trim_tm][k])
        elif name == 'phi_f_fr':
            for log2_F_phi_diff in p_f:
                if not isinstance(log2_F_phi_diff, int):
                    continue
                for pad_diff in p_f[log2_F_phi_diff]:
                    for sub in range(len(p_f[log2_F_phi_diff][pad_diff])):
                        p_f[log2_F_phi_diff][pad_diff][sub] = (
                            to_device(p_f[log2_F_phi_diff][pad_diff][sub]))
        else:  # no-cov
            raise ValueError("unknown filter name: %s" % name)

# misc #######################################################################
def _handle_paths_exclude(paths_exclude, j_all, n_psis, supported, names=None):
    """Handles `paths_exclude` argument of Scattering1D or
    TimeFrequencyScattering1D.

      - Ensures `paths_exclude` is dict and doesn't have unsupported keys
      - Ensures the provided n and j aren't out of bounds
      - Handles negative indexing
      - Handles `key: int` (expected `key: list[int]`)
      - "Converts" from j to n (fills all 'n' that have the specified 'j')
      - Doesn't handle `'n2, n1'`
    """
    # check basic structure
    if paths_exclude is None:  # no-cov
        paths_exclude = {nm: [] for nm in supported}
        return paths_exclude
    elif not isinstance(paths_exclude, dict):  # no-cov
        raise ValueError("`paths_exclude` must be dict, got %s" % type(
            paths_exclude))

    # fill what's missing as we can't change size of dict during iteration
    for nm in supported:
        if nm not in paths_exclude:
            paths_exclude[nm] = []

    # in case we handle some paths separately (n2, n1_fr in JTFS)
    if names is None:
        names = list(paths_exclude)

    for p_name in names:
        # # ensure all keys are functional
        assert p_name in supported, (p_name, supported)
        # ensure list
        if isinstance(paths_exclude[p_name], int):
            paths_exclude[p_name] = [paths_exclude[p_name]]
        else:
            try:
                paths_exclude[p_name] = list(paths_exclude[p_name])
            except:  # no-cov
                raise ValueError(("`paths_exclude` values must be list[int] "
                                  "or int, got paths_exclude['{}'] type: {}"
                                  ).format(p_name,
                                           type(paths_exclude[p_name])))

        # n2, n1_fr ######################################################
        if p_name[0] == 'n':
            for i, n in enumerate(paths_exclude[p_name]):
                # handle negative
                if n < 0:
                    paths_exclude[p_name][i] = n_psis + n

                # warn if 'n2' already excluded
                if p_name == 'n2':
                    n = paths_exclude[p_name][i]
                    n_j2_0 = [n2 for n2 in range(n_psis) if j_all[n2] == 0]
                    if n in n_j2_0:  # no-cov
                        warnings.warn(
                            ("`paths_exclude['n2']` includes `{}`, which "
                             "is already excluded (alongside {}) per "
                             "having j2==0."
                             ).format(n, ', '.join(map(str, n_j2_0))))
        # j2, j1_fr ######################################################
        elif p_name[0] == 'j':
            for i, j in enumerate(paths_exclude[p_name]):
                # handle negative
                if j < 0:
                    j = max(j_all) + j
                # forbid out of bounds
                if j > max(j_all):  # no-cov
                    raise ValueError(("`paths_exclude` exceeded maximum {}: "
                                      "{} > {}\nTo specify max j, use `-1`"
                                      ).format(p_name, j, max(j_all)))
                # warn if 'j2' already excluded
                elif p_name == 'j2' and j == 0:  # no-cov
                    warnings.warn(("`paths_exclude['j2']` includes `0`, "
                                   "which is already excluded."))

                # convert to n ###########################################
                # fetch all n that have the specified j
                n_j_all = [n for n in range(n_psis) if j_all[n] == j]

                # append if not already present
                n_name = 'n2' if p_name == 'j2' else 'n1_fr'
                for n_j in n_j_all:
                    if n_j not in paths_exclude[n_name]:
                        paths_exclude[n_name].append(n_j)

    return paths_exclude


def _handle_smart_paths(smart_paths, paths_exclude, psi1_f, psi2_f):
    if isinstance(smart_paths, (tuple, float)):
        if isinstance(smart_paths, tuple):
            kw = dict(e_loss=smart_paths[0], level=smart_paths[1])
        else:
            kw = dict(e_loss=smart_paths)
        assert 0 < kw['e_loss'] < 1, smart_paths
        paths_exclude_base = smart_paths_exclude(psi1_f, psi2_f, **kw)
    elif smart_paths == 'primitive':
        paths_exclude_base = primitive_paths_exclude(psi1_f, psi2_f)
    elif not smart_paths:
        paths_exclude_base = {'n2, n1': []}
    else:  # no-cov
        raise ValueError("`smart_paths` must be float, str['primitive'], dict, "
                         "or `False`, got %s" % str(smart_paths))
    paths_exclude.update(paths_exclude_base)
