import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

project = "RSR"
author = "Ji-Eun Byun, Hyeuk Ryu, Junho Song"
copyright = f"{datetime.now().year}, {author}"

try:
    from rsr import __version__ as _v
except Exception:
    _v = "0.0.0"
version = release = _v

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "torch",
    "ndtools_duco",
    "igraph",
    "networkx",
    "numpy",
    "matplotlib",
    "jsonschema",
    "yaml",
    "psutil",
    "typer",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

myst_enable_extensions = ["colon_fence", "deflist", "linkify"]
