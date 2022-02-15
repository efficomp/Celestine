# This file is part of source.
#
# Celestine is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# source is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# source. If not, see <http://www.gnu.org/licenses/>.
#
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

import os
import sys

__author__ = 'Juan Carlos Gómez-López'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.0'
__maintainer__ = 'Juan Carlos Gómez-López'
__email__ = 'goloj@ugr.es'
__status__ = 'Development'

sys.path.insert(0, os.path.abspath('../../source'))
sys.setrecursionlimit(1500)

# -- Project information -----------------------------------------------------

project = 'Celestine'
copyright = '2021, EFFICOMP'
author = 'Juan Carlos Gómez-López'

# The short X.Y version
# version = ''
# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Autodoc flags
autodoc_default_flags = ['members',
                         'undoc-members',
                         'private-members',
                         'special-members',
                         'inherited-members',
                         'show-inheritance']

# References to external modules
intersphinx_mapping = { 'python': ('https://docs.python.org/3', None),
                        'numpy': ('https://numpy.org/doc/stable/', None),
                        'pandas': ('https://pandas.pydata.org/docs/', None),
                        'sklearn': ('https://scikit-learn.org/stable/', None),
                        'pymongo': ('https://pymongo.readthedocs.io/en/stable/', None)}

# -- Options for HTML output -------------------------------------------------

html_title = 'source'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = './_static/efficomp_logo1.png'
html_favicon = './_static/efficomp_logo1.ico'