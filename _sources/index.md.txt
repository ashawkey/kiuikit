# Kiuikit

> NOTE: this doc is still under construction!

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
   :caption: API
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./mesh.md
   ./camera.md
   ./utils.md
   ./ops.md
   ./vis.md
   ./misc_api.md

.. toctree::
   :caption: Tool
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./render.md
   ./misc.md