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
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

# List of modules to be mocked up. Useful when some external dependencies are
# not met at build time and break the building process.
autodoc_mock_imports = ['tensorflow']


##### Custom scraper #########################################################
from glob import glob
import shutil
import os
from sphinx_gallery.scrapers import figure_rst


class PlotScraper(object):
    def __init__(self):
        self.seen = set()

    def __repr__(self):
        return 'PlotScraper'

    def __call__(self, block, block_vars, gallery_conf):
        # Find all PNG and mp4 files in the directory of this example.
        path_current_example = os.path.dirname(block_vars['src_file'])
        supported = ('png', 'jpg', 'svg', 'gif')
        files = [values for ext in supported for values in
                 sorted(glob(os.path.join(path_current_example, f'*.{ext}')))]

        # Iterate through PNGs, copy them to the sphinx-gallery output directory
        file_names = list()
        image_path_iterator = block_vars['image_path_iterator']
        for file in files:
            if file not in self.seen:
                self.seen |= set(file)
                this_path = image_path_iterator.next()
                this_path = this_path.replace(Path(this_path).suffix,
                                              Path(file).suffix)
                file_names.append(this_path)
                shutil.move(file, this_path)
        # Use the `figure_rst` helper function to generate rST for image files
        return figure_rst(file_names, gallery_conf['src_dir'])

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
html_css_files = ['style.css']

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
pygments_style = 'default'

# sphinx-gallery configuration
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': ['../../examples'],
    # path to where to save gallery generated output
    'gallery_dirs': ['examples-rendered'],
    # specify that examples should be ordered according to filename
    'within_subsection_order': FileNameSortKey,
    # yes
    'filename_pattern': '',
    # yes
    'reset_modules': (),
    # yes
    'run_stale_examples': False,
    # yes
    'matplotlib_animations': True,
    # yes
    'image_scrapers': ('matplotlib', PlotScraper()),
}

# configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/', None),
}

# for image scraper ##########################################################
import matplotlib as mpl
mpl.rcParams['savefig.bbox'] = 'tight'

# copy images over ###########################################################
docspath = confdir.parent
src_imgdir = Path(docspath, 'source', '_images')
build_imgdir = Path(docspath, 'build', 'html', '_images')
# make dir if doesn't exist
for d in (build_imgdir.parent.parent, build_imgdir.parent, build_imgdir):
    if not d.is_dir():
        os.mkdir(d)
# copy files
img_exts = ('.png', '.jpg', '.mp4', '.gif')
for file in src_imgdir.iterdir():
    if file.suffix in img_exts:
        shutil.copy(file, Path(build_imgdir, file.name))


#### Theme configs ##########################################################
# import sphinx_rtd_theme
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = 'alabaster'

html_theme_options = {
    'logo': 'favicon.png',
    'touch_icon': 'favicon.png',
    'logo_name': 'WaveSpin',
    'page_width': '70%',
    # 'description': ('Wavelet Scattering, Joint Time-Frequency Scattering: '
    #                 'features for audio, biomedical, and other applications, '
    #                 'in Python'),
    'description': 'Scattering Discriminative Invariants',
    'github_button': True,
    'github_type': 'star',
    'github_user': 'gptanon',
    'github_repo': 'wavespon',
    'github_banner': True,
    'fixed_sidebar': False,
    'font_family': 'Helvetica, Arial, sans-serif',
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
