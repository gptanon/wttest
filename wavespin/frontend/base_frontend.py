# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018- The Kymatio developers
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the Modified BSD License
# (BSD 3-clause; see NOTICE.txt in the WaveSpin root directory for details).
# -----------------------------------------------------------------------------
import importlib


class ScatteringBase():
    """
    This is a modification of `kymatio/frontend/base_frontend.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def __init__(self):
        super(ScatteringBase, self).__init__()

    def build(self):
        """Defines elementary routines.

        This function should always call and create the filters via
        self.create_filters() defined below. For instance, via:
        self.filters = self.create_filters()
        """
        raise NotImplementedError

    def _instantiate_backend(self, import_string):
        """This function should instantiate the backend to be used if not already
        specified
        """

        # Either the user entered a string, in which case we load the
        # corresponding backend.
        if isinstance(self.backend, str):
            if self.backend.startswith(self.frontend_name):
                try:
                    self.backend = importlib.import_module(
                        import_string + self.backend + "_backend",
                        'backend').backend
                except ImportError as e:  # no-cov
                    raise ImportError('Backend ' + self.backend + ' not found!\n'
                                      '%s' % str(e))
            else:  # no-cov
                raise ImportError('The backend ' + self.backend +
                                  ' can not be called from the frontend ' +
                                  self.frontend_name + '.')
        # Either the user passed a backend object, in which case we perform
        # a compatibility check.
        else:
            if not self.backend.name.startswith(self.frontend_name):
                raise ImportError('The backend ' + self.backend.name +
                                  ' is not supported by the frontend ' +
                                  self.frontend_name + '.')

    def create_filters(self):
        """This function should run a filterbank function that
        will create the filters as numpy array, and then, it should
        save those arrays.
        """
        raise NotImplementedError

    def gpu(self):
        """This function should handle each backend's GPU execution."""
        raise NotImplementedError

    def cpu(self):
        """This function should handle each backend's CPU execution."""
        raise NotImplementedError

    def to_device(self):
        """This function should handle each backend's execution on a
        custom device.
        """
        raise NotImplementedError

    def update_filters(self, name):
        """This function should handle each backend's backend-specific filter
        operations (device placement, module buffers, etc) for a specific
        filter set (e.g. `name='psi1_f'`).
        """
        raise NotImplementedError
