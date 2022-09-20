# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018- The Kymatio developers
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the Modified BSD License
# (BSD 3-clause; see NOTICE.txt in the WaveSpin root directory for details).
# -----------------------------------------------------------------------------
import importlib


class ScatteringEntry(object):
    """Entry point for all scattering objects.

    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/frontend/entry.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name')
        self.class_name = kwargs.pop('class_name')

        frontend_suffixes = {'torch' : 'Torch',
                             'numpy' : 'NumPy',
                             'tensorflow' : 'TensorFlow'}

        frontend = kwargs.pop('frontend', 'numpy').lower()
        supported = list(frontend_suffixes)
        if frontend not in supported:
            raise ValueError(("Frontend '{}' is not valid. Must be one of: {}"
                              ).format(frontend, ', '.join(supported)))

        try:
            module = importlib.import_module(
                "wavespin.{}.frontend.{}_frontend".format(
                    self.class_name, frontend))

            # Create frontend-specific class name by inserting frontend name
            # after `Scattering`.
            frontend = frontend_suffixes[frontend]

            class_name = self.__class__.__name__

            base_name = class_name[:-len('Entry*D')]
            dim_suffix = class_name[-len('*D'):]

            class_name = base_name + frontend + dim_suffix

            self.__class__ = getattr(module, class_name)
            self.__init__(*args, **kwargs)
        except Exception as e:
            raise e from RuntimeError(f"\nThe frontend '{frontend}' could not be "
                                      "correctly imported.")


__all__ = ['ScatteringEntry']
