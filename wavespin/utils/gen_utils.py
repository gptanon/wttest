# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import sys
import numpy as np
from scipy.fft import ifft
from copy import deepcopy
from pathlib import Path


def fill_default_args(cfg, defaults, copy_original=True,
                      check_against_defaults=False):
    """If a key is present in `defaults` but not in `cfg`, then copies
    the key-value pair from `defaults` onto `cfg`. Also applies to nests.

    `check_against_defaults` will raise Exception is there's keys in `cfg`
    that aren't in `defaults`.
    """
    # always copy defaults since they're assigned to `cfg` which may be modified
    defaults = deepcopy(defaults)

    # handle inputs
    if cfg is None or cfg == {}:
        return defaults
    elif not isinstance(cfg, dict):  # no-cov
        raise ValueError("`cfg` must be dict or None, got %s" % type(cfg))

    # don't affect external
    if copy_original:
        cfg = deepcopy(cfg)

    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
        else:
            if isinstance(v, dict):
                cfg[k] = fill_default_args(cfg[k], v)

    if check_against_defaults:
        for k in cfg:
            if k not in defaults:  # no-cov
                raise ValueError("unknown kwarg: '{}', supported are:\n{}".format(
                    k, '\n'.join(list(defaults))))
    return cfg

# misc #######################################################################
def is_real(xf, is_time=False):
    if not is_time:
        xf = ifft(xf)
    return bool(np.abs(xf.imag).max() <
                np.finfo(xf.dtype).eps * 10)


def append_to_sys_path(path):
    """Append `path` to `sys.path` unless it's already present."""
    path = Path(path)
    assert path.is_file() or path.is_dir(), str(path)
    if not any(str(path).lower() == p.lower() for p in sys.path):
        sys.path.insert(0, str(path))


def print_table(*cols, show=True):
    """Pretty-print like a table. Each arg is a separate column in the table.

    `show=False` to silently return the text that would be printed.

    Examples:

    ::

        values0 = [1, 2, 3]
        values1 = ['a', 'b', 'c']
        print_table(values0, values1)

        name_values0 = {'a': [1, 2]}
        name_values1 = {'b': [3, 4]}
        # name_values_invalid = {'a': [1, 2], 'b': [3, 4]}
        print_table(name_values0, name_values1)

    """
    # if any column has a header, fill the rest with empty if they don't.
    cols = list(cols)
    if any(isinstance(c, dict) for c in cols):
        has_headers = True
        for i in range(len(cols)):
            if not isinstance(cols[i], dict):
                cols[i] = {'': cols[i]}
    else:
        has_headers = False
        # enforce same format, but won't make headers
        cols = [{'': c} for c in cols]

    # convert all entries to string
    for ci, c in enumerate(cols):
        for header, rows in c.items():
            for ri in range(len(rows)):
                cols[ci][header][ri] = str(rows[ri])

    # ensure same number of rows; fill empty if not so
    max_n_rows = -1
    for c in cols:
        rows = list(c.values())[0]
        max_n_rows = max(max_n_rows, len(rows))
    for c in cols:
        rows = list(c.values())[0]
        while len(rows) < max_n_rows:
            rows.append('')

    # determine longest entry in each column
    longest_row_per_col = []
    for c in cols:
        longest_row_per_col.append(0)
        rows = list(c.values())[0]
        header = list(c)[0]
        for row in rows:
            longest_row_per_col[-1] = max(longest_row_per_col[-1],
                                          len(row), len(header))

    # make table, justified by `longest_row_per_col`
    txt = ""
    for ri in range(max_n_rows):
        row_txt = ""
        for ci, c in enumerate(cols):
            row = list(c.values())[0][ri]
            justify = longest_row_per_col[ci]
            row_str = row.ljust(justify)
            row_txt += "{} | ".format(row_str)
        row_txt = row_txt[:-2]  # drop the last `| `
        txt += row_txt + '\n'

    # handle header
    if has_headers:
        header_txt = ""
        for ci, c in enumerate(cols):
            header = list(c)[0]
            justify = longest_row_per_col[ci]
            header_str = header.ljust(justify)
            header_txt += "{} | ".format(header_str)
        header_txt = header_txt[:-2]  # drop the last `| `
        # underline header
        header_txt = "\033[4m" + header_txt + "\033[0m"

        txt = header_txt + '\n' + txt

    if show:
        print(txt)
    else:
        return txt


# backend ####################################################################
class ExtendedUnifiedBackend():
    """Extends existing WaveSpin backend with functionality.

    Handles all non-`core` uses. May be replaced in future by the usual backends.
    """
    def __init__(self, x_or_backend_name):
        if isinstance(x_or_backend_name, str):
            backend_name = x_or_backend_name
        else:
            backend_name = _infer_backend(x_or_backend_name, get_name=True)[1]
        if backend_name == 'jaxlib':  # no-cov
            backend_name = 'jax'  # standardize
        self.backend_name = backend_name

        if backend_name == 'torch':
            import torch
            self.B = torch
        elif backend_name == 'tensorflow':
            import tensorflow as tf
            self.B = tf
        else:
            self.B = np
        self.Bk = get_wavespin_backend(backend_name)

    def __getattr__(self, name):
        # fetch from wavespin.backend if possible
        if hasattr(self.Bk, name):
            return getattr(self.Bk, name)
        raise AttributeError(f"'{self.Bk.__name__}' object has no "
                             f"attribute '{name}'")  # no-cov

    def abs(self, x):
        return self.B.abs(x)

    def log(self, x):
        if self.backend_name in ('numpy', 'jax'):
            out = np.log(x)
        elif self.backend_name == 'torch':
            out = self.B.log(x)
        else:
            out = self.B.math.log(x)
        return out

    def sum(self, x, axis=None, keepdims=False):
        if self.backend_name in ('numpy', 'jax'):
            out = np.sum(x, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.sum(x, dim=axis, keepdim=keepdims)
        else:
            out = self.B.reduce_sum(x, axis=axis, keepdims=keepdims)
        return out

    def norm(self, x, ord=2, axis=None, keepdims=True):
        if self.backend_name in ('numpy', 'jax'):
            if ord == 1:
                out = np.sum(np.abs(x), axis=axis, keepdims=keepdims)
            elif ord == 2:
                out = np.linalg.norm(x, ord=None, axis=axis, keepdims=keepdims)
            else:
                out = np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.norm(x, p=ord, dim=axis, keepdim=keepdims)
        else:
            out = self.B.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        return out

    def median(self, x, axis=None, keepdims=None):
        if keepdims is None and self.backend_name != 'tensorflow':
            keepdims = True
        if self.backend_name in ('numpy', 'jax'):
            out = np.median(x, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.median(x, dim=axis, keepdim=keepdims)
            # torch may return `values` and `indices` if `axis is not None`
            if isinstance(out.values, self.B.Tensor):
                out = out.values
        else:
            if axis is not None or keepdims is not None:  # no-cov
                raise ValueError("`axis` and `keepdims` for `median` in "
                                 "TensorFlow backend are not implemented.")
            v = self.B.reshape(x, [-1])
            m = v.get_shape()[0]//2
            out = self.B.reduce_min(self.B.nn.top_k(v, m, sorted=False).values)
        return out

    def std(self, x, axis=None, keepdims=True):
        if self.backend_name in ('numpy', 'jax'):
            out = np.std(x, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.std(x, dim=axis, keepdim=keepdims)
        else:
            out = self.B.math.reduce_std(x, axis=axis, keepdims=keepdims)
        return out

    def min(self, x, axis=None, keepdims=False):
        if self.backend_name in ('numpy', 'jax'):
            out = np.min(x, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            kw = {'dim': axis} if axis is not None else {}
            if keepdims:
                kw['keepdim'] = True
                if axis is None:  # no-cov
                    # torch lacks multi-dim `dim` and we don't implement it here
                    raise NotImplementedError
            out = self.B.min(x, **kw)
        else:
            out = self.B.math.reduce_min(x, axis=axis, keepdims=keepdims)
        return out

    def numpy(self, x):
        if self.backend_name in ('numpy', 'jax'):
            out = x
        else:
            if hasattr(x, 'to') and 'cpu' not in x.device.type:
                x = x.cpu()
            if getattr(x, 'requires_grad', False):
                x = x.detach()
            out = x.numpy()
        return out


def npy(x):
    """Functional form of wavespin.toolkit.ExtendedUnifiedBackend.numpy()."""
    if isinstance(x, list):
        if not hasattr(x[0], 'ndim'):
            return np.array([float(_x) for _x in x])
        else:
            return np.array([npy(_x) for _x in x])
    elif isinstance(x, (float, int)):
        return x

    if not isinstance(x, np.ndarray):
        if hasattr(x, 'to') and 'cpu' not in x.device.type:
            x = x.cpu()
        if getattr(x, 'requires_grad', False):
            x = x.detach()
        if hasattr(x, 'numpy'):
            x = x.numpy()
        else:
            x = np.array(x)
    return x


def _infer_backend(x, get_name=False):
    while isinstance(x, (dict, list)):
        if isinstance(x, dict):
            if 'coef' in x:
                x = x['coef']
            else:
                x = list(x.values())[0]
        else:
            x = x[0]

    module = type(x).__module__.split('.')[0]

    if module == 'numpy':
        backend = np
    elif module == 'torch':
        import torch
        backend = torch
    elif module == 'tensorflow':
        import tensorflow
        backend = tensorflow
    elif module in ('jaxlib', 'jax'):
        import jax
        backend = jax
    elif isinstance(x, (int, float)):
        # list of lists, fallback to numpy
        module = 'numpy'
        backend = np
    else:  # no-cov
        raise ValueError("could not infer backend from %s" % type(x))
    return (backend if not get_name else
            (backend, module))


def get_wavespin_backend(backend_name):
    if backend_name == 'numpy':
        from ..backend.numpy_backend import NumPyBackend as B
    elif backend_name == 'torch':
        from ..backend.torch_backend import TorchBackend as B
    elif backend_name == 'tensorflow':
        from ..backend.tensorflow_backend import TensorFlowBackend as B
    elif backend_name in ('jaxlib', 'jax'):
        from ..backend.jax_backend import JaxBackend as B
    return B


def backend_has_gpu(backend_name):
    if backend_name == 'torch':
        import torch
        return torch.cuda.is_available()
    elif backend_name == 'tensorflow':
        import tensorflow as tf
        return bool(tf.config.list_physical_devices('GPU') != [])
    elif backend_name == 'jax':
        import jax
        try:
            _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
            return True
        except:
            return False
    elif backend_name == 'numpy':  # no-cov
        return False
    else:  # no-cov
        raise ValueError(f"unknown backend '{backend_name}'")
