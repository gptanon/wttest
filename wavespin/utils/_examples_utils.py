# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Utilities for running repository examples."""
import shutil
from pathlib import Path
from IPython.display import Image, display


def display_image(path, copy_to_pwd=False, set_retina_format=False):
    # show in IPython
    _ = display(Image(path))

    # copy to present working directory for personal inspection or for
    # documentation scrapers
    if copy_to_pwd:
        shutil.copy(path, Path(path).name)


def show_visual(viz_fn, prerendered_filename, prerendered_flag):
    path = Path('..', 'docs', 'source', '_images', 'examples',
                prerendered_filename).resolve()
    display_fn = lambda: display_image(path, copy_to_pwd=True)

    if prerendered_flag:
        try:
            display_fn()
        except:
            viz_fn()
    else:
        viz_fn()
