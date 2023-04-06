import os
import subprocess

name = 'nvblox'

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

subprocess.call('doxygen', shell=True)

html_logo = "images/nvblox_logo_128.png"

source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser',
}

extensions = [
    'sphinx.ext.autosectionlabel'
]

project = name
master_doc = 'root'

html_extra_path = ['doxyoutput/html']

source_suffix = ['.md']

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'
