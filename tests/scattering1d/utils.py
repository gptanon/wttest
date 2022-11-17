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
from wavespin.utils.gen_utils import append_to_sys_path

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


# special imports ############################################################
FUNCS_DIR = Path(Path(__file__).parent, 'funcs')
append_to_sys_path(FUNCS_DIR)
from funcs.scattering1d_legacy import scattering1d as scattering1d_legacy

# misc #######################################################################
# used to load saved coefficient outputs
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
assert os.path.isdir(TEST_DATA_DIR), TEST_DATA_DIR
assert len(os.listdir(TEST_DATA_DIR)) > 0, os.listdir(TEST_DATA_DIR)
