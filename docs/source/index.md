# Kiuikit

> NOTE: this doc is still unfinished!

A toolkit for computer vision (especially 3D vision) tasks.

## Features
* Collection of *maintained, reusable and trustworthy* code snippets.
* Always using lazy import so the code is not slowed down by `import kiui`.
* Useful CLI tools, such as a GUI mesh renderer.

## Installation
```bash
# released
pip install kiui # install the minimal package
pip install kiui[full] # install optional dependencies

# latest
pip install -U git+https://github.com/ashawkey/kiuikit.git
```

<!-- toctree -->

.. toctree::
   :caption: APIs
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./mesh.md
   ./camera.md
   ./utils.md
   ./ops.md

.. toctree::
   :caption: Tools
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./render.md
   ./misc.md