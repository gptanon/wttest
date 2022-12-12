# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018- The Kymatio developers
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the Modified BSD License
# (BSD 3-clause; see NOTICE.txt in the WaveSpin root directory for details).
# -----------------------------------------------------------------------------
import torch

class ScatteringTorch(torch.nn.Module):
    """
    This is a modification of `kymatio/frontend/torch_frontend.py` in
    https://github.com/kymatio/kymatio/blob/0.3.0/
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def __init__(self):
        super(ScatteringTorch, self).__init__()
        self.frontend_name = 'torch'

    def register_filters(self):
        """ This function should be called after filters are generated,
        saving those arrays as module buffers. """
        raise NotImplementedError

    def forward(self, x):
        """This method is an alias for `scattering`."""
        self.backend.input_checks(x)
        return self.scattering(x)

    _doc_array = 'torch.Tensor'

    _doc_alias_name = 'or `forward`'

    _doc_frontend_paragraph = \
        """
        This class inherits from `torch.nn.Module`. As a result, it has all
        the same capabilities, including transferring the object to the GPU
        using the `cuda` or `to` methods. This object would then take GPU
        tensors as input and output the scattering coefficients of those
        tensors.
        """
