# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# pylint: disable=redefined-builtin

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

from typing import List
import os
import sys

# Modify PYTHONPATH so we can obtain the version data from setup module.
# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nvblox_torch')))
from setup import NVBLOX_VERSION_NUMBER, NVBLOX_VERSION_PATCH

# NOTE(alexmillane, 2025-04-24): This file is in a seperate folder to avoid
# duplicate configuration errors coming from mypy. The only way I could find
# to solve this was to add this new folder.

# -- Project information -----------------------------------------------------

project = 'nvblox_torch'
copyright = '2025, NVIDIA'
author = 'NVIDIA'
released = True    # Indicates if this is a public or internal version of the repo.

# -- General configuration ---------------------------------------------------

sys.path.append(os.path.abspath('_ext'))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    'sphinx_tabs.tabs',
    'sphinx_copybutton',
    # TODO(alexmillane, 2025-04-24): Try re-enabling this once we have pydocs generating.
    #    'autodocsumm'
    'nvblox_torch_doc_tools'
]

# put type hints inside the description instead of the signature (easier to read)
autodoc_typehints = 'description'
# document class *and* __init__ methods
autoclass_content = 'both'    #

todo_include_todos = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Be picky about missing references
nitpicky = True    # warns on broken references
nitpick_ignore: List[str] = []    # can exclude known bad refs

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'nvidia_sphinx_theme'
html_title = 'nvblox'
html_show_sphinx = False
html_theme_options = {
    'copyright_override': {
        'start': 2023
    },
    'pygments_light_style': 'tango',
    'pygments_dark_style': 'monokai',
    'footer_links': {},
    'github_url': 'https://github.com/nvidia-isaac/nvblox',
    # TODO(alexmillane, 2025-04-24): Try re-enabling this once we have a pypi page.
    # "icon_links": [
    #     {
    #         "name": "PyPI",
    #         "url": "https://pypi.org/project/nvblox",
    #         "icon": "fa-brands fa-python",
    #         "type": "fontawesome",
    #     },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = []
html_static_path = ['_static']
html_css_files = ['custom.css']

# Todos
todo_include_todos = True

# Linkcheck
# NOTE(alexmillane, 2025-05-09): The links in the main example page are relative links
# which are only valid post-build. linkcheck doesn't like this. So here we ignore
# links to the example pages via html.
linkcheck_ignore = [
    r'pages/torch_examples_.*\.html',    # Ignore all pages/torch_examples_*.html links
]

#####################################
#  Macros dependent on release state
#####################################
# pylint: disable=line-too-long
nvblox_torch_docs_config = {
    'released': released,
    'internal_wheel_base_url': 'https://urm.nvidia.com/artifactory/hw-nvblox-alpine-local/' + \
        'pypi/release/nvblox_torch/',
    'external_wheel_base_url': 'https://github.com/nvidia-isaac/nvblox/releases' + \
        f'/download/v{NVBLOX_VERSION_NUMBER}',
    'wheel_name_ubuntu_24_cuda_12': \
        f'nvblox_torch-{NVBLOX_VERSION_NUMBER}{NVBLOX_VERSION_PATCH}+cu12ubuntu24-863-py3-none-linux_x86_64.whl',
    'wheel_name_ubuntu_22_cuda_12': \
        f'nvblox_torch-{NVBLOX_VERSION_NUMBER}{NVBLOX_VERSION_PATCH}+cu12ubuntu22-863-py3-none-linux_x86_64.whl',
    'wheel_name_ubuntu_22_cuda_11': \
        f'nvblox_torch-{NVBLOX_VERSION_NUMBER}{NVBLOX_VERSION_PATCH}+cu11ubuntu22-863-py3-none-linux_x86_64.whl',
    'wheel_name_ubuntu_24_cuda_13': \
        f'nvblox_torch-{NVBLOX_VERSION_NUMBER}{NVBLOX_VERSION_PATCH}+cu13ubuntu24-863-py3-none-linux_x86_64.whl',
    'internal_git_url': 'ssh://git@gitlab-master.nvidia.com:12051/nvblox/nvblox.git',
    'external_git_url': 'git@github.com:nvidia-isaac/nvblox.git',
    'internal_code_link_base_url': 'https://gitlab-master.nvidia.com/nvblox/nvblox/-/tree/main',
    'external_code_link_base_url': 'https://github.com/nvidia-isaac/nvblox/tree/public'
}
