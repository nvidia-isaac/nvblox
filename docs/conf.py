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

# NOTE: sphinx-multiversion doesn't provide a reliable way to detect the version
# at conf.py import time. We need to use the setup() function to access Sphinx config.
# For now, set a placeholder that will be updated in setup().
NVBLOX_VERSION_NUMBER = '0.0.8'    # Will be overridden in setup()
NVBLOX_VERSION_PATCH = 'rc5'    # For v0.0.8


# Define setup() function to properly detect version after Sphinx initializes
# pylint: disable=import-outside-toplevel,broad-exception-caught
def setup(app: object) -> None:
    """Sphinx setup function to detect version from sphinx-multiversion context.

    This function runs after Sphinx initializes and has access to app.srcdir,
    which points to the correct source directory for each version being built.
    """
    import re
    import subprocess

    def _get_wheel_name(version: str, ubuntu: str, cuda: str) -> str:
        """Generate wheel filename based on version."""
        version_patches = {
            '0.0.8': 'rc5',
            '0.0.9': '.dev1',
        }
        patch = version_patches.get(version, '.dev1')
        return f'nvblox_torch-{version}{patch}+cu{cuda}ubuntu{ubuntu}-863-py3-none-linux_x86_64.whl'

    def _update_version_config(version: str) -> None:
        """Update all version-dependent configuration values."""
        global NVBLOX_VERSION_NUMBER
        NVBLOX_VERSION_NUMBER = version
        app.config.html_title = f'nvblox_torch {NVBLOX_VERSION_NUMBER}'

        # Update wheel URLs and names in nvblox_torch_docs_config
        app.config.nvblox_torch_docs_config['external_wheel_base_url'] = \
            f'https://github.com/nvidia-isaac/nvblox/releases/download/v{version}'
        app.config.nvblox_torch_docs_config['wheel_name_ubuntu_24_cuda_12'] = \
            _get_wheel_name(version, '24', '12')
        app.config.nvblox_torch_docs_config['wheel_name_ubuntu_22_cuda_12'] = \
            _get_wheel_name(version, '22', '12')
        app.config.nvblox_torch_docs_config['wheel_name_ubuntu_22_cuda_11'] = \
            _get_wheel_name(version, '22', '11')
        app.config.nvblox_torch_docs_config['wheel_name_ubuntu_24_cuda_13'] = \
            _get_wheel_name(version, '24', '13')

    # Try to get version from various sources
    # 1. Check environment variable (sphinx-multiversion should set this)
    smv_current_version = os.environ.get('SPHINX_MULTIVERSION_NAME')
    if smv_current_version:
        match = re.match(r'v?(\d+\.\d+\.\d+)', smv_current_version)
        if match:
            _update_version_config(match.group(1))
            return

    # 2. Try git in the source directory
    try:
        result = subprocess.run(['git', 'describe', '--all', '--exact-match', 'HEAD'],
                                capture_output=True,
                                text=True,
                                cwd=app.srcdir,
                                check=False)
        if result.returncode == 0:
            ref_name = result.stdout.strip()
            match = re.search(r'v?(\d+\.\d+\.\d+)', ref_name)
            if match:
                _update_version_config(match.group(1))
                return
    except Exception:
        pass    # Git detection failed, continue to fallback

    # 3. Fallback: read from setup.py in the source directory
    # This is the most reliable method for sphinx-multiversion builds
    setup_path = os.path.join(app.srcdir, '..', 'nvblox_torch', 'setup.py')
    try:
        with open(setup_path, 'r', encoding='utf-8') as f:
            content = f.read()
        match = re.search(r"NVBLOX_VERSION_NUMBER\s*=\s*['\"]([^'\"]+)['\"]", content)
        if match:
            _update_version_config(match.group(1))
    except Exception:
        pass    # Use default version


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
    'sphinx_multiversion',
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
html_title = f'nvblox_torch {NVBLOX_VERSION_NUMBER}'
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
smv_remote_whitelist = r'^.*$'
smv_branch_whitelist = r'^(public|v0.0.8-docs|v0.0.9-docs)$'
smv_tag_whitelist = r'^(v0.0.8|v0.0.9)$'
html_sidebars = {'**': ['versioning.html', 'sidebar-nav-bs']}

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
