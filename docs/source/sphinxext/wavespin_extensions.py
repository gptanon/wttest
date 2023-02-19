# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
#
"""Sphinx extensions for WaveSpin."""

##### Custom section titles ##################################################
# e.g.
#
#   Examples
#   --------

def scrape_titles_in_folder(_dir):
    def scrape_titles_in_file(p):
        with open(p, 'r') as f:
            txt_lines = f.readlines()
        titles = []
        inside_of_docstring = False
        for i, line in enumerate(txt_lines):
            if '"""' in line:
                inside_of_docstring = not inside_of_docstring
            # '---' or shorter is prone to false positives
            subsection_symbols = ('-', '^')  # can add more

            if (inside_of_docstring and
                    # `3` or less is prone to false positives (e.g. '-')
                    any(s * 4 in line for s in subsection_symbols)):
                contender = txt_lines[i - 1].strip(' \n')
                if contender != '':  # can appear for some reason
                    titles.append(contender)

            # e.g. """Docstring."""
            if line.count('"""') == 2:
                inside_of_docstring = not inside_of_docstring
        return titles

    all_titles = []
    for p in _dir.iterdir():
        if p.suffix == '.py':
            all_titles.extend(scrape_titles_in_file(p))
        elif p.is_dir():
            all_titles.extend(scrape_titles_in_folder(p))
    return list(set(all_titles))  # unique only


##### Custom scraper #########################################################
import os
import shutil
from pathlib import Path
from glob import glob
from sphinx_gallery.scrapers import figure_rst


class PlotScraper(object):
    def __init__(self):
        self.seen = set()

    def __repr__(self):
        return 'PlotScraper'

    def __call__(self, block, block_vars, gallery_conf):
        # Find all supported files in the directory of this example.
        path_current_example = os.path.dirname(block_vars['src_file'])
        supported = ('png', 'jpg', 'svg', 'gif')
        sort_key = os.path.getmtime  # sort by last generated
        files = [values for ext in supported for values in
                 sorted(glob(os.path.join(path_current_example, f'*.{ext}')),
                        key=sort_key)
                 ]

        # Iterate through files, copy them to the sphinx-gallery output directory
        file_names = list()
        # srcsetpaths = list()
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


# ensure images are included #################################################
import textwrap

def silently_include_images(confdir):
    def append_images(txt, imgdir, save_relpath):
        img_exts = ('.png', '.jpg', '.mp4', '.gif')
        for file in imgdir.iterdir():
            if file.suffix in img_exts:
                txt += """
                       .. image:: {}/{}
                         :height: 0px
                         :width: 0px
                       """.format(save_relpath, file.name)
        return txt

    docspath = confdir.parent
    src_imgdir = Path(docspath, 'source', '_images')

    # get image paths & make .rst text
    txt = ""
    txt = append_images(txt, src_imgdir, '_images')
    # also handle `internal`
    internal_imgdir = Path(docspath, 'source', '_images', 'internal')
    txt = append_images(txt, internal_imgdir, '_images/internal')

    # unindent multiline string
    txt = textwrap.dedent(txt)

    # make a .txt to be `.. include`-ed; this is so that `<img src=` works
    with open('silent_image_includes.txt', 'w') as f:
        f.write(txt)

    # make `_examples_gallery_indented.txt` from `_examples_gallery.txt`
    with open('_examples_gallery.txt', 'r') as f:
        loaded = f.read()

    with open('_examples_gallery_indented.txt', 'w') as f:
        new = loaded
        for url in ('<img src="', '<a href="'):
            new = new.replace(url, url.replace('"', '"../'))
        f.write(new)

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
