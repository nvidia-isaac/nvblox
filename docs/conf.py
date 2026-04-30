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

import datetime
import shutil
import subprocess
from typing import List
import os
import sys

# Modify PYTHONPATH so we can import the helpers module.
# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath('.'))
from helpers import TemporaryLinkcheckIgnore, to_datetime, is_expired

# NOTE(alexmillane, 2025-04-24): This file is in a seperate folder to avoid
# duplicate configuration errors coming from mypy. The only way I could find
# to solve this was to add this new folder.

# -- Project information -----------------------------------------------------

project = 'nvblox_torch'
copyright = '2025, NVIDIA'
author = 'NVIDIA'
released = False    # Indicates if this is a public or internal version of the repo.

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
    'sphinx_multiversion',
    'breathe',
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
nitpick_ignore_regex = [
    # External C/C++ types not in our Doxygen XML.
    (r'cpp:.*', r'(u?int\d+_t|size_t|float|double|bool|void)'),
    (r'cpp:.*', r'Vk\w+'),    # Vulkan types
    (r'cpp:.*', r'Eigen(::\w+)*'),    # Eigen types
    (r'cpp:.*', r'CudaStream'),
    (r'cpp:.*', r'(Depth|Color)Image'),
    (r'cpp:.*', r'ColorMesh'),
    (r'cpp:.*', r'Camera'),
    (r'cpp:.*', r'VkWindow(::\w+)*'),
    (r'cpp:.*', r'BufferedVisualizer\b.*'),
    (r'cpp:.*', r'TexturedVisualizer\b.*'),
    (r'cpp:.*', r'nvblox(::\w+)*'),
    (r'cpp:.*', r'SharedTexture(::\w+)*'),
    (r'cpp:.*', r'SharedBuffer'),
    (r'cpp:.*', r'BaseVisualizer'),
    (r'cpp:.*', r'ViewCamera'),
    (r'cpp:.*', r'PipelineBuilder'),
    (r'ref\..*', r'.*_8h_source'),    # Doxygen file-source labels
    (r'ref', r'.*_8h_source'),    # Same, without sub-role
]

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

# Versioning (sphinx-multiversion)
# Regexp for public and release branches: vX.Y.WZ
smv_remote_whitelist = r'^.*$'
smv_branch_whitelist = r'^(public|v\d+\.\d+\.\d+)$'
smv_tag_whitelist = r'^(v\d+\.\d+\.\d+)$'
html_sidebars = {'**': ['versioning.html', 'sidebar-nav-bs']}

# Todos
todo_include_todos = True

# Linkcheck
# NOTE(alexmillane, 2025-05-09): The links in the main example page are relative links
# which are only valid post-build. linkcheck doesn't like this. So here we ignore
# links to the example pages via html.
linkcheck_ignore = [
    r'pages/torch_examples_.*\.html',    # Ignore all pages/torch_examples_*.html links
    r'pages/core_library_.*\.html',    # Ignore all pages/core_library_*.html links
]

temporary_linkcheck_ignore = [
    TemporaryLinkcheckIgnore(
        url='https://3dmatch.cs.princeton.edu/',
        start_date=to_datetime('09.07.2025'),
        days=14,
    ),
    TemporaryLinkcheckIgnore(
        url='https://sun3d.cs.princeton.edu/',
        start_date=to_datetime('09.07.2025'),
        days=14,
    ),
]

for ignore in temporary_linkcheck_ignore:
    if not is_expired(ignore.start_date, ignore.days):
        print(f'Ignoring {ignore.url} until '
              f'{ignore.start_date + datetime.timedelta(days=ignore.days)}')
        linkcheck_ignore.append(ignore.url)

#####################################
#  Macros dependent on release state
#####################################

nvblox_torch_docs_config = {
    'released': released,
    'internal_git_url': 'ssh://git@gitlab-master.nvidia.com:12051/nvblox/nvblox.git',
    'external_git_url': 'git@github.com:nvidia-isaac/nvblox.git',
    'internal_code_link_base_url': 'https://gitlab-master.nvidia.com/nvblox/nvblox/-/blob/main',
    'external_code_link_base_url': 'https://github.com/nvidia-isaac/nvblox/blob/public'
}

#####################################
#  Doxygen / Breathe (C++ API docs)
#####################################

_docs_dir = os.path.abspath(os.path.dirname(__file__))
_doxygen_xml = os.path.join(_docs_dir, '_build', 'doxygen', 'xml')

if shutil.which('doxygen'):
    os.makedirs(_doxygen_xml, exist_ok=True)
    subprocess.run(['doxygen', 'Doxyfile'], cwd=_docs_dir, check=False)
else:
    print('WARNING: doxygen not found — C++ API docs will be incomplete.')

breathe_projects = {'nvblox_renderer': _doxygen_xml}
breathe_default_project = 'nvblox_renderer'
breathe_default_members = ('members', )
