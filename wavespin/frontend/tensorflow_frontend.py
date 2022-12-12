# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018- The Kymatio developers
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the Modified BSD License
# (BSD 3-clause; see NOTICE.txt in the WaveSpin root directory for details).
# -----------------------------------------------------------------------------
import tensorflow as tf

class ScatteringTensorFlow(tf.Module):
    """
    This is a modification of `kymatio/frontend/tensorflow_frontend.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def __init__(self, name):
        super(ScatteringTensorFlow, self).__init__(name=name)
        self.frontend_name = 'tensorflow'

    @tf.Module.with_name_scope
    def __call__(self, x):
        """This method is an alias for `scattering`."""
        return self.scattering(x)

    _doc_array = 'tf.Tensor'

    _doc_alias_name = ''

    _doc_frontend_paragraph = \
        """
        This class inherits from `tf.Module`. As a result, it has all the
        same capabilities as a standard TensorFlow `Module`.
        """
