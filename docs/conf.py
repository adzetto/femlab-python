import os
import sys
import inspect

sys.path.insert(0, os.path.abspath("../src"))

project = "femlabpy"
copyright = "2026, femlabpy contributors"
author = "femlabpy contributors"

# The full version, including alpha/beta/rc tags
release = "0.6.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",
    "numpydoc",
    "myst_parser",
]


# GitHub linkcode resolution
def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    # Make path relative to the root src directory
    try:
        rel_fn = os.path.relpath(fn, start=os.path.abspath("../src"))
        # Clean path separators for URL
        rel_fn = rel_fn.replace("\\", "/")
    except ValueError:
        return None

    return f"https://github.com/adzetto/femlabpy/blob/main/src/{rel_fn}{linespec}"


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
    "Algorithm",
]
