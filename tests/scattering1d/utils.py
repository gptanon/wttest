# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Methods reused in testing."""
import os, contextlib, tempfile, shutil, warnings, inspect
from pathlib import Path
import numpy as np
from wavespin.utils.gen_utils import append_to_sys_path
from wavespin import toolkit as tkt

# should be `1` before committing
FORCED_PYTEST = 1
# tests to skip
SKIPS = {
  'jtfs': 0,
  'visuals': 0,
  'toolkit': 0,
  'long_in_jtfs': 0,
}

# helpers ####################################################################
def cant_import(backend_name):
    if backend_name == 'numpy':
        return False
    elif backend_name == 'torch':
        try:
            import torch
        except ImportError:
            warnings.warn("Failed to import torch")
            return True
    elif backend_name == 'tensorflow':
        try:
            import tensorflow
        except ImportError:
            warnings.warn("Failed to import tensorflow")
            return True
    elif backend_name == 'jax':
        try:
            import jax
        except ImportError:
            warnings.warn("Failed to import jax")
            return True
    else:  # no-cov
        raise ValueError(f"Unknown backend '{backend_name}'")


def get_wavespin_backend(backend_name):
    if backend_name == 'numpy':
        from wavespin.scattering1d.backend.numpy_backend import NumPyBackend1D
        return NumPyBackend1D
    elif backend_name == 'torch':
        from wavespin.scattering1d.backend.torch_backend import TorchBackend1D
        return TorchBackend1D
    elif backend_name == 'tensorflow':
        from wavespin.scattering1d.backend.tensorflow_backend import (
            TensorFlowBackend1D)
        return TensorFlowBackend1D


@contextlib.contextmanager
def tempdir(dirpath=None):
    if dirpath is not None and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
        os.mkdir(dirpath)
    elif dirpath is None:
        dirpath = tempfile.mkdtemp()
    else:
        os.mkdir(dirpath)
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


def add_method_with_line_replaced(obj, method_name, method_name_new,
                                  to_replace, replaced_by, imports_code=''):
    code = inspect.getsource(getattr(obj, method_name))
    lines = code.split('\n')

    # fetch first line
    line0 = lines[0]
    # ensure we begin with definition
    assert "def" in line0, line0

    # figure out indent to unindent
    if not line0.startswith('def'):
        indent = len(line0) - len(line0.lstrip(' '))
        for i, line in enumerate(lines):
            lines[i] = line[indent:]
    line0 = lines[0]

    # rename
    lines[0] = line0.replace(method_name, method_name_new)

    # replace line of interest
    for i, line in enumerate(lines):
        if to_replace in line:
            lines[i] = line.replace(to_replace, replaced_by)
    code_new = '\n'.join(lines)
    if imports_code != '':
        code_new = imports_code + '\n' + code_new

    # execute and assign
    exec(code_new)
    fn_new = eval(f'{method_name_new}')
    setattr(obj, method_name_new, fn_new.__get__(obj))


class IgnoreWarnings(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        warnings.filterwarnings("ignore", message=f".*{self.message}.*")

    def __exit__(self, *_):
        warnings.filterwarnings("default", message=f".*{self.message}.*")


ignore_pad_warnings = IgnoreWarnings("boundary effects")

# test helpers ###############################################################
def run_meta_tests_jtfs(jtfs, Scx, jmeta, test_params, extras=False):
    def assert_equal_lengths(Scx, jmeta, field, pair, out_3D, test_params_str,
                             jtfs):
        """Assert that number of coefficients and frequency rows for each match"""
        if out_3D:
            out_n_coeffs  = len(Scx[pair])
            out_n_freqs   = sum(len(c['coef'][0]) for c in Scx[pair])
            meta_n_coeffs = len(jmeta[field][pair])
            meta_n_freqs  = np.prod(jmeta[field][pair].shape[:2])

            assert out_n_coeffs == meta_n_coeffs, (
                "len(out[{0}]), len(jmeta[{1}][{0}]) = {2}, {3}\n{4}"
                ).format(pair, field, out_n_coeffs, meta_n_coeffs,
                         test_params_str)
        else:
            out_n_freqs  = sum(c['coef'].shape[1] for c in Scx[pair])
            meta_n_freqs = len(jmeta[field][pair])

        assert out_n_freqs == meta_n_freqs, (
            "out vs meta n_freqs mismatch for {}, {}: {} != {}\n{}".format(
                pair, field, out_n_freqs, meta_n_freqs, test_params_str))

    def assert_equal_values(Scx, jmeta, field, pair, i, meta_idx, out_3D,
                            test_params_str, test_params, jtfs):
        """Assert that non-NaN values are equal."""
        a, b = Scx[pair][i][field], jmeta[field][pair][meta_idx[0]]
        errmsg = ("(out[{0}][{1}][{2}], jmeta[{2}][{0}][{3}]) = ({4}, {5})\n{6}"
                  ).format(pair, i, field, meta_idx[0], a, b, test_params_str)

        meta_len = b.shape[-1] if field != 'spin' else 1
        zeroth_order_unaveraged = bool(pair == 'S0' and
                                       not test_params['average'])
        if field not in ('spin', 'stride'):
            assert meta_len == 3, ("all meta fields (except spin, stride) must "
                                   "pad to length 3: %s" % errmsg)
            if not zeroth_order_unaveraged:
                assert len(a) > 0, ("all computed metas (except spin) must "
                                    "append something: %s" % errmsg)

        if field == 'stride':
            assert meta_len == 2, ("'stride' meta length must be 2 "
                                   "(got meta: %s)" % b)
            if pair in ('S0', 'S1'):
                if pair == 'S1' or test_params['average']:
                    assert len(a) == 1, errmsg
                if pair == 'S0' and not test_params['average']:
                    assert a == (), errmsg
                    assert np.all(np.isnan(b)), errmsg
                else:
                    assert a == b[..., 1], errmsg
                    assert np.isnan(b[..., 0]), errmsg
            else:
                assert len(a) == 2, errmsg
                assert np.all(a == b), errmsg
                assert not np.any(np.isnan(b)), errmsg

        elif (field == 'spin' and pair in ('S0', 'S1')
              ) or zeroth_order_unaveraged:
            assert len(a) == 0 and np.all(np.isnan(b)), errmsg

        elif len(a) == meta_len:
            assert np.all(a == b), errmsg

        elif len(a) < meta_len:
            # S0 & S1 have one meta entry per coeff so we pad to joint's len
            if np.all(np.isnan(b[:2])):
                assert pair in ('S0', 'S1'), errmsg
                assert a[0] == b[..., 2], errmsg
            # joint meta is len 3 but at compute 2 is appended
            elif len(a) == 2 and meta_len == 3:
                assert pair not in ('S0', 'S1'), errmsg
                assert np.all(a[:2] == b[..., :2]), errmsg

        else:
            # must meet one of above behaviors
            raise AssertionError(errmsg)

        # increment meta_idx for next check (manual 'key')
        meta_idx_prev = meta_idx[0]  # store for ref
        if pair in ('S0', 'S1') or out_3D:
            meta_idx[0] += 1
        else:
            # increment by number of frequential rows (i.e. `n1`) since
            # these n1-meta aren't appended in computation
            n_freqs = Scx[pair][i]['coef'].shape[1]
            meta_idx[0] += n_freqs

        # check 'key'
        if pair in ('S0', 'S1'):
            n_freqs_coeff = 1
        else:
            n_freqs_coeff = Scx[pair][i]['coef'].shape[1]
        key = jmeta['key'][pair][i]
        expected = (meta_idx_prev if not out_3D else
                    meta_idx_prev * n_freqs_coeff)  # since (n_slices, n_freqs, t)
        assert key.start == expected, (key.start, expected, pair, i)
        n_freqs_key = key.stop - key.start
        assert n_freqs_key == n_freqs_coeff, (n_freqs_key, n_freqs_coeff, pair, i)

    def assert_aligned_stride(Scx, test_params_str, jtfs):
        """Assert all frequential strides are equal in `aligned`."""
        ref_stride = Scx['psi_t * psi_f_up'][0]['stride'][0]
        for pair in Scx:
            # skip non second order
            if pair in ('S0', 'S1'):
                continue
            # 'phi_f' exempt from alignment in this case
            elif 'phi_f' in pair and (not jtfs.average_fr and
                                      jtfs.scf.average_fr_global_phi):
                continue
            for i, c in enumerate(Scx[pair]):
                s = c['stride'][0]
                assert s == ref_stride, (
                    "Scx[{}][{}]['stride'] = {} != ref_stride == {}\n{}".format(
                        pair, i, s, ref_stride, test_params_str))

    test_params_str = '\n'.join(f'{k}={v}' for k, v in test_params.items())

    # ensure no output shape was completely reduced
    for pair in Scx:
        for i, c in enumerate(Scx[pair]):
            assert not np.any(c['coef'].shape == 0), (pair, i)

    # meta test
    out_3D = test_params['out_3D']
    for field in ('j', 'n', 'spin', 'stride'):
      for pair in jmeta[field]:
        assert_equal_lengths(Scx, jmeta, field, pair, out_3D,
                             test_params_str, jtfs)
        meta_idx = [0]
        for i in range(len(Scx[pair])):
            assert_equal_values(Scx, jmeta, field, pair, i, meta_idx,
                                out_3D, test_params_str, test_params, jtfs)

    # check stride if `aligned`
    if jtfs.aligned:
        assert_aligned_stride(Scx, test_params_str, jtfs)

    # Bundled tests: save re-compute and test these ##########################
    # this one's cheap so always do it
    _ = jtfs.info(show=False)

    if extras:
        if jtfs.average:
            for structure in (1, 2, 3, 4):
                for separate_lowpass in (False, True):
                    try:
                        _ = tkt.pack_coeffs_jtfs(
                            Scx, jmeta, structure=structure,
                            separate_lowpass=separate_lowpass,
                            sampling_psi_fr=jtfs.scf.sampling_psi_fr,
                            out_3D=jtfs.out_3D)
                    except Exception as e:
                        print(test_params_str)
                        raise e


# special imports ############################################################
FUNCS_DIR = Path(Path(__file__).parent, 'funcs')
append_to_sys_path(FUNCS_DIR)
from funcs.scattering1d_legacy import scattering1d as scattering1d_legacy

# misc #######################################################################
# used to load saved coefficient outputs
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
assert os.path.isdir(TEST_DATA_DIR), TEST_DATA_DIR
assert len(os.listdir(TEST_DATA_DIR)) > 0, os.listdir(TEST_DATA_DIR)
