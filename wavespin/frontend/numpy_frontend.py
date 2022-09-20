# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018- The Kymatio developers
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the Modified BSD License
# (BSD 3-clause; see NOTICE.txt in the WaveSpin root directory for details).
# -----------------------------------------------------------------------------

class ScatteringNumPy:
    """
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/frontend/
    numpy_frontend.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def __init__(self):
        self.frontend_name = 'numpy'

    def __call__(self, x):
        """This method is an alias for `scattering`."""
        self.backend.input_checks(x)
        return self.scattering(x)

    _doc_array = 'np.ndarray'

    _doc_alias_name = ''

    _doc_frontend_paragraph = ''
