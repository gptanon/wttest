# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the Modified BSD License
# (BSD 3-clause; see NOTICE.txt in the WaveSpin root directory for details).
# -----------------------------------------------------------------------------
from .numpy_frontend import ScatteringNumPy

class ScatteringJax(ScatteringNumPy):
    def __init__(self):
        # all else stands identical
        self.frontend_name = 'jax'
