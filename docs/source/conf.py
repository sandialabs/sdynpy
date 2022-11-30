# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../../src'))


def get_version():
    VERSIONFILE = os.path.join('..', '..', 'src', 'sdynpy', '__init__.py')
    with open(VERSIONFILE, 'rt') as f:
        lines = f.readlines()
    vgx = '^__version__ = \"[0-9+.0-9+.0-9+]*[a-zA-Z0-9]*\"'
    for line in lines:
        mo = re.search(vgx, line, re.M)
        if mo:
            return mo.group().split('"')[1]
    raise RuntimeError('Unable to find version in %s.' % (VERSIONFILE,))


project = 'SDynPy'
author = 'Daniel P. Rohe'
copyright = '2022 National Technology & Engineering Solutions of Sandia, \
    LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, \
    the U.S. Government retains certain rights in this software'

# The full version, including alpha/beta/rc tags
release = get_version()
version = release
print(release)


templates_path = ['_templates']
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme = 'sphinx_rtd_theme'
html_theme_options = {'navigation_depth': 8}
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'matplotlib.sphinxext.plot_directive',
    'sphinxcontrib.bibtex',
    'nbsphinx',
    'sphinx_copybutton'
]
# autodoc_mock_imports = ["sdynpy"]
latex_engine = 'xelatex'
bibtex_bibfiles = ['main.bib']
bibtex_default_style = 'plain'
plot_html_show_formats = False
plot_html_show_source_link = False
plot_include_source = True
add_module_names = False
plot_rcparams = {'font.size': 10}
plot_formats = [('png', 300)]
