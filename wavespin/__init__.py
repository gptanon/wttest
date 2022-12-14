# -*- coding: utf-8 -*-
"""
MIT License
===========

Some wavespin source files under other terms (see NOTICE.txt).

Copyright (c) 2022 John Muradeli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


__version__ = '0.1.2'
__title__ = 'wavespin'
__author__ = 'John Muradeli'
__license__ = __doc__
__copyright__ = 'Copyright (c) 2022, %s.' % __author__
__project_url__ = 'https://github.com/OverLordGoldDragon/wavespin'


from .configs import CFG
from .scattering1d import (
    ScatteringEntry1D as Scattering1D,
    TimeFrequencyScatteringEntry1D as TimeFrequencyScattering1D)
from .scattering1d.refining import smart_paths_exclude
from . import toolkit
from .toolkit import fit_smart_paths
from .scattering1d.filter_bank import morlet_1d, gauss_1d
