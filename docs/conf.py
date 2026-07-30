"""Sphinx configuration for pyQSC_JAX."""

from pyqsc_jax import __version__

project = "pyQSC_JAX"
author = "UW Plasma"
copyright = "2026, UW Plasma"
version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_title = f"pyQSC_JAX {version}"
intersphinx_mapping = {
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "python": ("https://docs.python.org/3/", None),
}
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath"]
