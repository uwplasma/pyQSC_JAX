"""Sphinx configuration for the pyQSC_JAX documentation."""

import datetime
import json
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent

project = "pyQSC_JAX"
author = "UWPlasma, University of Wisconsin-Madison"
copyright = f"{datetime.date.today().year}, UWPlasma"

# Document the source tree next to this file, not whatever copy is installed.
sys.path.insert(0, str(DOCS.parent / "src"))
from pyqsc_jax import __version__ as release  # noqa: E402

version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "scripts/*", "examples/output/*"]

myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath", "substitution"]
myst_heading_anchors = 3
myst_dmath_double_inline = True


_SUPERSCRIPT = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


def _format(value):
    """Render ``3.1e-14`` as ``3.1 × 10⁻¹⁴``; other values unchanged."""
    text = f"{value:g}" if isinstance(value, float) else str(value)
    match = re.fullmatch(r"(-?[0-9.]+)e([+-]?)0*([0-9]+)", text)
    if match:
        mantissa, sign, exponent = match.groups()
        exponent = ("-" if sign == "-" else "") + exponent
        return f"{mantissa} \u00d7 10{exponent.translate(_SUPERSCRIPT)}"
    return text


# Numbers quoted in the text come from the same runs that made the figures
# (docs/scripts/measurements.py and fig_examples.py).
myst_substitutions = {
    key: _format(value)
    for key, value in json.loads(
        (DOCS / "_static" / "figures" / "measurements.json").read_text()
    ).items()
    if not isinstance(value, dict)
}

autodoc_default_options = {"members": True}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_use_rtype = False

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

html_theme = "pydata_sphinx_theme"
html_title = "pyQSC_JAX"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_theme_options = {
    "github_url": "https://github.com/uwplasma/pyQSC_JAX",
    "icon_links": [
        {"name": "PyPI", "url": "https://pypi.org/project/pyqsc-jax/", "icon": "fa-brands fa-python"}
    ],
    "header_links_before_dropdown": 7,
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": [],
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}
html_context = {
    "github_user": "uwplasma",
    "github_repo": "pyQSC_JAX",
    "github_version": "main",
    "doc_path": "docs",
}
html_sidebars = {"index": []}
