# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'kiui'
copyright = '2024, kiui'
author = 'kiui'

# The full version, including alpha/beta/rc tags
release = '0.2.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "m2r2",
]

# Pull documentation types from hints
autodoc_typehints = "both"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True, # will doc all methods and attributes
    "member-order": "bysource",
    "undoc-members": True, # also doc memebers without a docstring
    "special-members": "__init__", # doc __init__ too
    "inherited-members": False,
    "imported-members": False,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_title = "kiuikit"
html_theme_options = {
    "light_css_variables": {
        "color-code-background": "#f4f4f4",
        "color-code-foreground": "#000",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ashawkey/kiuikit",
            "class": "",
        },
    ],
    "light_logo": "logo.jpg",
    "dark_logo": "logo.jpg",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

sphinx_to_github = True
sphinx_to_github_verbose = True
sphinx_to_github_encoding = "utf-8"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True