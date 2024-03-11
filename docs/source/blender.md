# Blender

To render complex 3D models (specifically with multiple submeshes), the only choice is to use `blender` and `bpy`.

We provide an example code for rendering with blender:

```bash
python -m kiui.cli.blender_render --help
```

Its features include:
* Set which GPU to use with `--gpu 0`.
* Render with random HDRI environment texture (check `assets/blender_lights`).
* Empirical cleaning of the scene (remove the annoying plane under the object).