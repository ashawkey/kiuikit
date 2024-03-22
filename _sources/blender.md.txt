# Blender Scripts

To render complex 3D models (specifically with multiple submeshes), the only choice is to use `blender` and `bpy`.

We provide an example code for rendering with blender:

```bash
# make sure you are using python>=3.10, tested with bpy==4.0.0
pip install bpy mathutils

# help
python -m kiui.cli.blender_render --help

# example
python -m kiui.cli.blender_render --mesh chest.glb --gpu 0 --depth --normal --albedo --camera --blend
```

Features include:
* Set which GPU to use with `--gpu 0` for `CYCLES` rendering engine.
* Render with random built-in HDRI environment texture shading (check `assets/blender_lights`).
* Optionally render depth (`exr`), normal, and albedo.
* Empirical cleaning of the scene (remove the annoying plane under the object).