import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "femlabpy"
copyright = "2026, femlabpy contributors"
author = "femlabpy contributors"

# The full version, including alpha/beta/rc tags
release = "0.6.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST setup
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
]

# Theme setup (PyData)
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/adzetto/femlabpy",
    "show_nav_level": 2,
    "navbar_align": "content",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
}

# Auto API generation
autosummary_generate = True
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

# Allow custom sections in numpydoc
numpydoc_custom_sections = [
    "Mathematical Formulation",
]
