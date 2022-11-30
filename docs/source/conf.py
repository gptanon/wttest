# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------

# Configuration file for the Sphinx documentation builder.

#### Path setup ##############################################################

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

# `docs/source` and `docs/` dirs, help Python find them if not already found
confdir = Path(__file__).parent
paths = [str(confdir), str(confdir.parent), str(Path(confdir, 'sphinxext'))]
for path in paths:
    path = Path(path)
    assert path.is_file() or path.is_dir(), str(path)
    if not any(str(path).lower() == p.lower() for p in sys.path):
        sys.path.insert(0, str(path))

#### Project info  ###########################################################
import wavespin
from sphinx_gallery.sorting import FileNameSortKey

project = 'WaveSpin'
author = wavespin.__author__
copyright = wavespin.__copyright__

# The short X.Y version
version = wavespin.__version__
# The full version, including alpha/beta/rc tags
release = wavespin.__version__

#### General configs #########################################################

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_toolbox.collapse',
    'texext',
    'm2r2',
]

# custom extensions
from wavespin_extensions import (
    scrape_titles_in_folder,
    PlotScraper,
    silently_include_images,
    PrettyPrintIterable,
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

# List of modules to be mocked up. Useful when some external dependencies are
# not met at build time and break the building process.
autodoc_mock_imports = ['tensorflow']

##### Custom section titles ##################################################
module_dir = Path(confdir.parent.parent, 'wavespin')
section_titles = scrape_titles_in_folder(module_dir)

napoleon_custom_sections = section_titles

##### HTML output configs ####################################################
import re
from sphinx_gallery.sorting import ExplicitOrder

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# CSS to customize HTML output
html_css_files = ['style.css']

# credit: https://logo.com
html_logo = '_images/favicon.png'
html_favicon = '_images/favicon.png'

# Make "footnote [1]_" appear as "footnote[1]_"
trim_footnote_reference_space = True

# ReadTheDocs sets master doc to index.rst, whereas Sphinx expects it to be
# contents.rst:
master_doc = 'index'

# make `code` code, instead of ``code``
default_role = 'literal'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# sphinx-gallery configuration
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': ['../../examples/'],
    # path to where to save gallery generated output
    'gallery_dirs': ['examples-rendered'],
    # specify that examples should be ordered according to filename
    'within_subsection_order': FileNameSortKey,
    # all `.py`
    'filename_pattern': '',
    # don't build subdirectories (inside of `examples/`)
    'ignore_pattern': r'{0}examples{0}.*{0}.*'.format(re.escape(os.sep)),
    # disable per using custom `rcParams`
    'reset_modules': (),
    # don't re-run unless source `.py` changes
    'run_stale_examples': False,
    # needed for animations
    'matplotlib_animations': True,
    # pick up animations produced by scripts, also non-`plt.figure()` plots
    'image_scrapers': ('matplotlib', PlotScraper()),
    # DPI options for browsers to support high and low DPI screens
    'image_srcset': ['1x', '1.8x'],
    # don't pre-insert %matplotlib inline (already done in `wavespin.visuals`)
    'first_notebook_cell': None,
    # don't gallery `more/` & others, as they're by definition script-only
    'nested_sections': False,
    # sort by priority rather than alphabetically
    'subsection_order': ExplicitOrder(['../../examples/more',
                                       '../../examples/internal']),
}

# configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/', None),
}

# for image scraper
import matplotlib as mpl
mpl.rcParams['savefig.bbox'] = 'tight'

# ensure images are included
silently_include_images(confdir)

#### Theme configs ##########################################################
# html_theme = 'alabaster'
html_theme = 'sphinx_book_theme'

html_theme_options = {
    'logo_only': True,
    'show_toc_level': 2,
    'repository_url': 'https://github.com/gptanon/wttest/',
    # add a "link to repository" button
    'use_repository_button': True,
    # 'logo': 'favicon.png',
    # 'touch_icon': 'favicon.png',
    # 'logo_name': 'WaveSpin',
    # 'page_width': '70%',
    # 'description': 'Scattering Discriminative Invariants',
    # 'github_button': True,
    # 'github_type': 'star',
    # 'github_user': 'gptanon',
    # 'github_repo': 'wavespon',
    # 'github_banner': True,
    # 'fixed_sidebar': False,
    # 'font_family': 'Helvetica, Arial, sans-serif',
}

#### Autodoc configs ########################################################
# Document module / class methods in order of writing (rather than alphabetically)
autodoc_member_order = 'bysource'

def skip(app, what, name, obj, would_skip, options):
    # include private methods (but not magic) if they have documentation
    if name.startswith('_') and getattr(obj, '__doc__', '') and (
            '__%s__' % name.strip('__') != name):
        return False
    return would_skip

###############################################################################

def setup(app):
    app.add_css_file("style.css")
    app.add_directive('pprint', PrettyPrintIterable)
    app.connect("autodoc-skip-member", skip)
