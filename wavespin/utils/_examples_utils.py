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


def display_image(path, copy_to_pwd=False, set_retina_format=False,
                  use_display=False):
    # copy to present working directory for personal inspection or for
    # documentation scrapers
    if copy_to_pwd:
        shutil.copy(path, Path(path).name)

    # show in IPython
    if use_display:
        # if in a loop, Jupyter only shows last return value, so show explicitly
        _ = display(Image(path))
    else:
        # else don't use `display` as it'll print if it can't show image
        return Image(path)


def show_visual(viz_fn, prerendered_filename, prerendered_flag):
    path = Path('..', 'docs', 'source', '_images', 'examples',
                prerendered_filename).resolve()
    display_fn = lambda: display_image(path, copy_to_pwd=True, use_display=False)

    if prerendered_flag:
        try:
            o = display_fn()
        except:
            o = viz_fn()
    else:
        o = viz_fn()
    return o
