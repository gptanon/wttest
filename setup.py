# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------

"""
WaveSpin
========

Joint Time-Frequency Scattering, Wavelet Scattering: features for audio,
biomedical, and other applications, in Python

WaveSpin features scattering transform implementations that maximize accuracy,
flexibility, and speed. Included are visualizations and convenience utilities
for transform and coefficient introspection and debugging.
"""

import os
import re
import warnings
from numpy import get_include as numpy_get_include
from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from Cython.Build import cythonize

current_path = os.path.abspath(os.path.dirname(__file__))


def read_file(*parts):
    with open(os.path.join(current_path, *parts), encoding='utf-8') as reader:
        return reader.read()


def get_requirements(*parts):
    with open(os.path.join(current_path, *parts), encoding='utf-8') as reader:
        return list(map(lambda x: x.strip(), reader.readlines()))


def find_version(*file_paths):
    version_file = read_file(*file_paths)
    version_matched = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                version_file, re.M)
    if version_matched:
        return version_matched.group(1)
    raise RuntimeError('Unable to find version')


#Set up the machinery to install custom fonts.  Subclass the setup tools install
#class in order to run custom commands during installation.
# class move_ttf(install):
#     def run(self):
#         """
#         Performs the usual install process and then copies the True Type fonts
#         that come with clearplot into matplotlib's True Type font directory,
#         and deletes the matplotlib fontList.cache
#         """
#         #Perform the usual install process
#         install.run(self)
#         # run only if matplotlib's installed
#         try:
#             import matplotlib as mpl
#         except ImportError:
#             1/0  # TODO
#             return

#         #Try to install custom fonts
#         import os, shutil
#         #Find where matplotlib stores its True Type fonts
#         mpl_data_dir = os.path.dirname(mpl.matplotlib_fname())
#         mpl_ttf_dir = os.path.join(mpl_data_dir, 'fonts', 'ttf')

#         #Copy the font files to matplotlib's True Type font directory
#         #(I originally tried to move the font files instead of copy them,
#         #but it did not seem to work, so I gave up.)
#         pkg_ttf_dir = os.path.join(os.path.dirname(__file__),
#                                    'wavespin', 'utils', '_fonts')
#         for file_name in os.listdir(pkg_ttf_dir):
#             if file_name.endswith('.ttf'):
#                 old_path = os.path.join(pkg_ttf_dir, file_name)
#                 new_path = os.path.join(mpl_ttf_dir, file_name)
#                 shutil.copyfile(old_path, new_path)
#                 print("Copied {} -> {}".format(old_path, new_path))

#         #Try to delete matplotlib's fontList cache
#         mpl_cache_dir = mpl.get_cachedir()
#         mpl_cache_dir_ls = os.listdir(mpl_cache_dir)

#         from pathlib import Path
#         from matplotlib.font_manager import FontManager
#         fm_path = Path(
#             mpl.get_cachedir(), f"fontlist-v{FontManager.__version__}.json")
#         os.remove(fm_path)
#         raise Exception("\n{}\n{}\{}".format(
#             fm_path, mpl.get_cachedir(), mpl_ttf_dir))
#         print("YES YES", fm_path)

#         if 'fontList.cache' in mpl_cache_dir_ls:
#             fontList_path = os.path.join(mpl_cache_dir, 'fontList.cache')
#             os.remove(fontList_path)
#             print("Deleted the matplotlib fontList.cache")
#         # except:
#         #     1/0
#             # warnings.warn("An issue occured while installing custom fonts for "
#             #               "wavespin.")


# Cython extensions
ext_modules = cythonize(Extension("wavespin.utils._compiled._algos",
                                  ["wavespin/utils/_compiled/_algos.pyx"]),
                        language_level=3)


setup(
    name="WaveSpin",
    version=find_version('wavespin', '__init__.py'),
    packages=find_packages(exclude=['tests', 'examples']),
    url="https://github.com/OverLordGoldDragon/wavespin",
    license="MIT",
    author="John Muradeli",
    author_email="john.muradeli@gmail.com",
    description=("Joint Time-Frequency Scattering, Wavelet Scattering: features "
                 "for audio, biomedical, and other applications, in Python"),
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    keywords=(
        "scattering-transform wavelets signal-processing visualization "
        "pytorch tensorflow python"
    ),
    install_requires=get_requirements('requirements.txt'),
    # setup_requires=["wheel", "setuptools>=18.0", "cython>=0.29.0"],  # TODO
    python_requires=">=3.7",
    tests_require=["pytest>=4.0", "pytest-cov"],
    ext_modules=ext_modules,
    include_dirs=[numpy_get_include()],
    include_package_data=True,
    zip_safe=True,
    # cmdclass={'install': move_ttf},
    #Specify any non-python files to be distributed with the package
    package_data = {'' : ['utils/_fonts/*.ttf']},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Utilities",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
