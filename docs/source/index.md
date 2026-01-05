# Kiuikit

> We try to keep the code simple and readable, so you are always encouraged to directly read the code!

A niche toolkit for computer vision (especially 3D vision) tasks.

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
   ./video.md
   ./equirect.md
   ./misc_api.md

.. toctree::
   :caption: Tool
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./sys.md
   ./render.md
   ./blender.md
   ./remesh.md
   ./agent.md
   ./slurm.md
   ./misc_cli.md