# Configuration file for the Sphinx documentation builder.

#### Path setup ##############################################################

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path


confdir = Path(__file__).parent
paths = [
    str(confdir),             # conf.py dir
    str(confdir.parents[0]),  # docs dir
    # str(confdir.parents[1]),  # package rootdir
]
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

    # 'nbsphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'texext',
    'm2r2',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


##### HTML output configs ####################################################

html_sidebars = { '**': [
    'about.html',
    'globaltoc.html',
    'relations.html',
    'searchbox.html'
] }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# CSS to customize HTML output
html_css_files = [
    'style.css',
]

# credit: https://logo.com
html_favicon = '_images/favicon.ico'

# Make "footnote [1]_" appear as "footnote[1]_"
trim_footnote_reference_space = True

# ReadTheDocs sets master doc to index.rst, whereas Sphinx expects it to be
# contents.rst:
master_doc = 'index'

# make `code` code, instead of ``code``
default_role = 'literal'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# sphinx-gallery configuration
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': ['../../examples'],
    # path to where to save gallery generated output
    'gallery_dirs': ['examples-rendered'],
    # specify that examples should be ordered according to filename
    'within_subsection_order': FileNameSortKey,
    # directory where function granular galleries are stored
    'backreferences_dir': 'gen_modules/backreferences',
    # Modules for which function level galleries are created.  In
    # this case sphinx_gallery and numpy in a tuple of strings.
    'doc_module': ('wavespin'),
    # yes
    'filename_pattern': '',
    # yes
    'reset_modules': (),
    # yes
    'run_stale_examples': False,
    # yes
    'matplotlib_animations': True,
}

# configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/', None),
}

# figure saving
import matplotlib as mpl
mpl.rcParams['savefig.bbox'] = 'tight'


#### Theme configs ##########################################################
# import sphinx_rtd_theme
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = 'alabaster'

html_theme_options = {
    'logo': 'favicon.png',
    'touch_icon': 'favicon.png',
    'logo_name': 'WaveSpin',
    # 'description': ('Wavelet Scattering, Joint Time-Frequency Scattering: '
    #                 'features for audio, biomedical, and other applications, '
    #                 'in Python'),
    'description': 'Wavelets and shite',
    'github_button': True,
    'github_type': 'star',
    'github_user': 'gptanon',
    'github_repo': 'wavespon',
    'fixed_sidebar': False,
    # 'font_family': '"Avenir Next", Avenir, "Helvetica Neue",Helvetica,Arial,sans-serif'
}

#### Autodoc configs ########################################################
from importlib import import_module
from docutils.parsers.rst import Directive
from docutils import nodes
from sphinx import addnodes
from inspect import getsource


# document lists, tuples, dicts, np.ndarray's exactly as in source code
class PrettyPrintIterable(Directive):
    required_arguments = 1

    def run(self):
        def _get_iter_source(src, varname):
            # 1. identifies target iterable by variable name, (cannot be spaced)
            # 2. determines iter source code start & end by tracking brackets
            # 3. returns source code between found start & end
            start = end = None
            open_brackets = closed_brackets = 0
            for i, line in enumerate(src):
                if line.startswith(varname):
                    if start is None:
                        start = i
                if start is not None:
                    open_brackets   += sum(line.count(b) for b in "([{")
                    closed_brackets += sum(line.count(b) for b in ")]}")

                if open_brackets > 0 and (open_brackets - closed_brackets == 0):
                    end = i + 1
                    break
            return '\n'.join(src[start:end])

        module_path, member_name = self.arguments[0].rsplit('.', 1)
        src = getsource(import_module(module_path)).split('\n')
        code = _get_iter_source(src, member_name)

        literal = nodes.literal_block(code, code)
        literal['language'] = 'python'

        return [addnodes.desc_name(text=member_name),
                addnodes.desc_content('', literal)]


# Document module / class methods in order of writing (rather than alphabetically)
autodoc_member_order = 'bysource'

def skip(app, what, name, obj, would_skip, options):
    # include private methods (but not magic) if they have documentation
    if name.startswith('_') and getattr(obj, '__doc__', '') and (
            '__%s__' % name.strip('__') != name):
        return False
    return would_skip

#### nbsphinx configs ###############################################
# nbsphinx_thumbnails = {
#     'examples/misc/timeseries': '_images/ecg.png',
# }


###############################################################################

def setup(app):
    app.add_css_file("style.css")
    app.add_directive('pprint', PrettyPrintIterable)
    app.connect("autodoc-skip-member", skip)
