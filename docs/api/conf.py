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
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../../src'))

def get_version():
    VERSIONFILE = os.path.join('..', '..', 'pyproject.toml')
    with open(VERSIONFILE, 'rt') as f:
        lines = f.readlines()
    vgx = 'version = \'[0-9+.0-9+.0-9+]*[a-zA-Z0-9]*\''
    for line in lines:
        mo = re.search(vgx, line, re.M)
        if mo:
            return mo.group().split("'")[1]
    raise RuntimeError('Unable to find version in %s.' % (VERSIONFILE,))

project = 'SDynPy'
copyright = '2022, Sandia National Laboratories'
author = 'Daniel P. Rohe'

# The full version, including alpha/beta/rc tags
release = get_version()
version = release
print(release)

templates_path = ['_templates']
# html_static_path = ['_static']
# html_css_files = ['custom.css']
html_theme = "sphinx_book_theme"
html_title = "SDynPy API Documentation v" + version
html_theme_options = {
    # 'navigation_depth': 8,
    # 'logo_only': True,
    # 'display_version':False
    "repository_url": "https://github.com/sandialabs/sdynpy",
    "use_repository_button": True,
}
html_logo = "../book/images/logo_horizontal.svg"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
# autodoc_mock_imports = ["sdynpy"]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_include_source = True
add_module_names = False
# plot_rcparams = {'font.size': 10}
# plot_formats = [('png', 300)]
