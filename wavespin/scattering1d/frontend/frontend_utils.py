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
from types import FunctionType
from ..refining import smart_paths_exclude, primitive_paths_exclude
from ...utils.gen_utils import backend_has_gpu, ExtendedUnifiedBackend


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
    # TODO move to reactive check?
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

    if self.frontend_name != 'numpy' and self.filters_device is None:
        self.cpu()

    if self.frontend_name == 'torch':
        # convert input to tensor if it isn't already
        p_ref = get_p_ref()
        device = p_ref.device.type
        if numpy_input:
            x = torch.as_tensor(x, device=device)
        elif x.device.type != device:
            x = x.to(device)

    elif self.frontend_name in ('tensorflow', 'jax'):
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
        return (
            isinstance(x, int) or
            (hasattr(x, 'dtype') and np.issubdtype(x.dtype, np.integer))
        )

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


def _check_jax_double_precision():
    import jax
    if jax.numpy.ones(5, dtype='float64').dtype.name != 'float64':
        raise Exception("Double precision with Jax backend requires "
                        "at-startup setup; refer to Jax docs.")

def _raise_reactive_setter(name, reactives_set_name, level_name='self'):
    raise AttributeError(
        f"`{name}` shouldn't be set as `{level_name}.{name}=value`. "
        f"Use `self.update({name}=value)` instead.\n"
        "Note that if this exception isn't raised, it doesn't mean the "
        f"non-`update` syntax is safe; see `self.{reactives_set_name}`")


def _setattr_and_handle_reactives(self, name, value, reactives):
    if name in reactives:
        setattr(self, f'_{name}', value)
    else:
        setattr(self, name, value)


def _warn_boundary_effects(diff, J_pad=None, min_to_pad=None, N=None, fr=False):
    do_warn = True
    if fr:
        vars_txt0 = "`J_fr` or `F`"
        vars_txt1 = "`max_pad_factor_fr`"
    else:
        vars_txt0 = "`J` or `T`"
        vars_txt1 = "`max_pad_factor`"

    if diff == 1:
        if J_pad is not None:
            ideal_pad = N + min_to_pad
            will_pad = 2**J_pad
            if ideal_pad / will_pad < 1.03:
                do_warn = False
        extent_txt = "Boundary"
    elif diff == 2:
        extent_txt = "Severe boundary"
    else:
        extent_txt = "Extreme boundary"

    if do_warn:
        extra_txt = "(not advised) "
        if J_pad is not None:
            pad_is_p2up = bool(J_pad == np.ceil(np.log2(N)))
            if pad_is_p2up:  # max_pad_factor == 0
                extra_txt = ""

        warnings.warn(f"{extent_txt} effects and filter distortion "
                      "expected per insufficient temporal padding; "
                      f"try lowering {vars_txt0}, or {extra_txt}"
                      f"increasing {vars_txt1}.")
        return True
    return False


def _handle_pad_mode(self):
    """Does NOT fully account for `pad_mode`."""
    supported = ('reflect', 'zero')
    if isinstance(self.pad_mode, FunctionType):
        _pad_fn = self.pad_mode

        def pad_fn(x):
            return _pad_fn(x, self.pad_left, self.pad_right)

        self._pad_mode = 'custom'

    elif self.pad_mode not in supported:  # no-cov
        raise ValueError(("unsupported `pad_mode` '{}';\nmust be a "
                          "function, or string, one of: {}"
                          ).format(str(self.pad_mode), ', '.join(supported)))

    else:
        def pad_fn(x):
            return self.backend.pad(x, self.pad_left, self.pad_right,
                                    self.pad_mode)
    self.pad_fn = pad_fn


def _handle_pad_mode_fr(self):
    """Does NOT fully account for `pad_mode_fr`."""
    supported = ('conj-reflect-zero', 'zero')
    if isinstance(self.pad_mode_fr, FunctionType):
        pad_fn_fr = self.pad_mode_fr

        self._pad_mode_fr = 'custom'

    elif self.pad_mode_fr not in supported:  # no-cov
        raise ValueError(("unsupported `pad_mode_fr` '{}';\nmust be a "
                          "function, or string, one of: {}").format(
                              str(self.pad_mode_fr), ', '.join(supported)))

    else:
        pad_fn_fr = None  # handled in `core`
    self.pad_fn_fr = pad_fn_fr

# device handling ############################################################
def _to_device(self, device=None, names=None):
    # make `to_device` to apply to each filter ###############################
    if self.frontend_name == 'torch':
        # unlike with other frontends, here we don't move filters, rather we
        # register them under `torch.nn.Module`, which then moves them to the
        # appropriate device at runtime via `load_device()` after `self.cuda()`
        # or `self.cpu()` has been called. See `register_filters()` in
        # `wavespin.scattering1d.frontend.torch_frontend.ScatteringTorch1D`.
        import torch

        if (isinstance(device, str) and device.lower() == 'gpu' and
                not backend_has_gpu('torch')):
            raise Exception("PyTorch could not find a GPU.")

        def to_device(x, fullname):
            x = torch.from_numpy(x)
            self.register_buffer(fullname, x)
            return x

    elif self.frontend_name == 'tensorflow':
        import tensorflow as tf

        if isinstance(device, str) and device.lower() in ('cpu', 'gpu'):
            devices = tf.config.list_physical_devices(device.upper())
            if devices == []:  # no-cov
                raise Exception("TensorFlow could not find a %s" % device.upper())
            device = devices[0].name.replace('physical_device:', '')

        def to_device(x, fullname):
            with tf.device(device):
                x = tf.identity(x, name=fullname)
            return x

    elif self.frontend_name == 'jax':
        import jax

        if isinstance(device, str) and device.lower() in ('cpu', 'gpu'):
            try:
                device = jax.devices(device)[0]
            except:  # no-cov
                raise Exception("Jax could not find a %s" % device.upper())

        def to_device(x, fullname):
            return jax.device_put(x, device=device)

    # move filters ###########################################################
    is_jtfs = bool(hasattr(self, 'scf'))

    if names is not None and not isinstance(names, (list, tuple)):  # no-cov
        names = [names]
    names_tm = ('phi_f', 'psi1_f', 'psi2_f', 'psi1_f_stacked')
    names_fr = ('phi_f_fr', 'psi1_f_fr_up', 'psi1_f_fr_dn',
                'psi1_f_fr_stacked_dict')
    if is_jtfs:
        if names is None:
            _move_filters_to_device_jtfs(self, to_device, names_tm)
            _move_filters_to_device_jtfs(self.scf, to_device, names_fr)
        else:
            for name in names:
                obj = (self if name in names_tm else
                       self.scf)
                _move_filters_to_device_jtfs(obj, to_device, [name])
    else:
        if names is None:
            names = names_tm
        _move_filters_to_device(self, to_device, names)

    self._moved_to_device = True


def _move_filters_to_device(self, to_device, names):
    if 'phi_f' in names:
        for k in self.phi_f:
            if isinstance(k, int):
                self.phi_f[k] = to_device(self.phi_f[k], _tensor_name('phi_f', k))

    if 'psi1_f' in names and not self.vectorized_early_U_1:
        for n1, psi_f in enumerate(self.psi1_f):
            for k in psi_f:
                if isinstance(k, int):
                    psi_f[k] = to_device(psi_f[k], _tensor_name('psi1_f', n1, k))

    if 'psi2_f' in names:
        for n2, psi_f in enumerate(self.psi2_f):
            for k in psi_f:
                if isinstance(k, int):
                    psi_f[k] = to_device(psi_f[k], _tensor_name('psi2_f', n2, k))

    if 'psi1_f_stacked' in names and self.vectorized_early_U_1:
        self.psi1_f_stacked = to_device(self.psi1_f_stacked,
                                        _tensor_name('psi1_f_stacked'))


def _tensor_name(attribute_name, *keys):
    return "t_{}__{}".format(attribute_name, '_'.join(map(str, keys)))


def _move_filters_to_device_jtfs(obj, to_device, filter_names):
    for name in filter_names:
        p_f = getattr(obj, name)

        if name == 'psi1_f_stacked':
            if obj.vectorized_early_U_1:
                obj.psi1_f_stacked = to_device(p_f, _tensor_name(name))

        elif name == 'psi1_f_fr_stacked_dict':
            if obj.vectorized_fr:
                for stack_id in p_f:
                    if obj.vectorized_early_fr:
                        p_f[stack_id] = to_device(
                            p_f[stack_id], _tensor_name(name, stack_id))
                    else:
                        for n1_fr_subsample in p_f[stack_id]:
                            p_f[stack_id][n1_fr_subsample] = to_device(
                                p_f[stack_id][n1_fr_subsample],
                                _tensor_name(name, stack_id, n1_fr_subsample)
                            )

        elif name in ('psi1_f', 'psi2_f'):
            if (name == 'psi2_f' or
                (name == 'psi1_f' and not obj.vectorized_early_U_1)):
                for n_tm in range(len(p_f)):
                    for k in p_f[n_tm]:
                        if not isinstance(k, int):
                            continue
                        p_f[n_tm][k] = to_device(
                            p_f[n_tm][k], _tensor_name(name, n_tm, k))

        elif name.startswith('psi') and 'fr' in name:
            if not obj.vectorized_fr:
                for psi_id in p_f:
                    if not isinstance(psi_id, int):
                        continue
                    for n1_fr in range(len(p_f[psi_id])):
                        p_f[psi_id][n1_fr] = to_device(
                            p_f[psi_id][n1_fr], _tensor_name(name, psi_id, n1_fr)
                        )

        elif name == 'phi_f':
            for trim_tm in p_f:
                if not isinstance(trim_tm, int):
                    continue
                for k in range(len(p_f[trim_tm])):
                    _nm = _tensor_name(name, trim_tm, k)
                    p_f[trim_tm][k] = to_device(p_f[trim_tm][k], _nm)

        elif name == 'phi_f_fr':
            for log2_F_phi_diff in p_f:
                if not isinstance(log2_F_phi_diff, int):
                    continue
                for pad_diff in p_f[log2_F_phi_diff]:
                    for sub in range(len(p_f[log2_F_phi_diff][pad_diff])):
                        _nm = _tensor_name(name, log2_F_phi_diff, pad_diff, sub)
                        p_f[log2_F_phi_diff][pad_diff][sub] = to_device(
                            p_f[log2_F_phi_diff][pad_diff][sub], _nm)

        else:  # no-cov
            raise ValueError("unknown filter name: %s" % name)


def _handle_device_non_filters_jtfs(self):
    [Dearly, DL] = [self.compute_graph_fr[k] for k in ('Dearly', 'DL')]
    B = ExtendedUnifiedBackend(self.backend.__name__.lower().split('backend')[0])

    # energy correction ------------------------------------------------------
    dtype = ('float32' if self.precision == 'single' else
             'float64')
    kw = dict(dtype=dtype, device=self.filters_device)

    # `Dearly`
    for pair in Dearly:
        if not isinstance(pair, dict):
            continue
        ec = Dearly[pair].get('energy_correction', None)
        if ec is not None:
            if isinstance(ec, list):
                for i, _ec in enumerate(ec):
                    Dearly[pair]['energy_correction'][i] = B.as_tensor(_ec, **kw)
            else:
                Dearly[pair]['energy_correction'] = B.as_tensor(ec, **kw)

    # `DL`
    for n2 in DL:
        for group_spec in DL[n2]:
            for key in DL[n2][group_spec]:
                ec = DL[n2][group_spec][key].get('energy_correction', None)
                if ec is not None:
                    DL[n2][group_spec][key]['energy_correction'
                                            ] = B.as_tensor(ec, **kw)

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


def _handle_smart_paths(smart_paths, paths_exclude, psi1_f, psi2_f, r_psi):
    if isinstance(smart_paths, (tuple, float)):
        if isinstance(smart_paths, tuple):
            kw = dict(e_loss=smart_paths[0], level=smart_paths[1])
        else:
            kw = dict(e_loss=smart_paths)
        assert 0 < kw['e_loss'] < 1, smart_paths
        paths_exclude_base = smart_paths_exclude(psi1_f, psi2_f, **kw,
                                                 r_psi=r_psi)
    elif smart_paths == 'primitive':
        paths_exclude_base = primitive_paths_exclude(psi1_f, psi2_f)
    elif not smart_paths:
        paths_exclude_base = {'n2, n1': []}
    else:  # no-cov
        raise ValueError("`smart_paths` must be float, str['primitive'], dict, "
                         "or `False`, got %s" % str(smart_paths))
    paths_exclude.update(paths_exclude_base)
