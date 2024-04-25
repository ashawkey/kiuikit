# Mesh Renderer

[[source]](https://github.com/ashawkey/kiuikit/blob/main/kiui/render.py)

We provide a convenient mesh renderer based on:
* [kiui.mesh.Mesh](mesh.md): our mesh implementation.
* [nvdiffrast](https://github.com/NVlabs/nvdiffrast): an NVIDIA GPU is required, install with `pip install git+https://github.com/NVlabs/nvdiffrast`.
* [dearpygui](https://github.com/hoffstadt/DearPyGui): **a desktop is required (cannot be forwarded by ssh)**. For headless servers, you can still use `--wogui` for rendering images and videos.


### Usage
```bash
# invoke renderer
kire --help 
# this is short for
python -m kiui.render --help
```

### Examples
```bash
## open a GUI to render a mesh
kire mesh.obj
kire mesh.obj --force_cuda_rast # if you cannot use OpenGL backend (usually happens for headless servers), this will fallback to CUDA backend with some limitations (ref: https://nvlabs.github.io/nvdiffrast/#rasterizing-with-cuda-vs-opengl-new)
kire mesh.obj --H 800 --W 800 --ssaa 2 # set resolution and use super-sampling anti-aliasing

kire mesh.obj --front_dir +x # specify mesh front-facing dir, default is +z (OpenGL convention). You can use [+-][xyz] to specify axis, and [123] to specify clockwise rotation (per 90 degree).

kire mesh.glb --pbr # render with PBR (metallic + roughness)
kire mesh.glb --pbr --envmap env.hdr # specify hdr file

## we can also run without GUI on headless servers
kire mesh.obj --save_video out.mp4 --wogui # save 360 degree rotating video
kire mesh.obj --save out --wogui # save rendered images to the out folder (controlled by --elevation and --num_azimuth)
```

In the GUI, you can use:
* `Left Drag` to rotate camera.
* `Middle Scroll` to scale camera.
* `Right Drag` to pan camera.
* `Space` to toggle rendering mode.
* `P` to toggle camera orbital rotation.
* `L` to toggle plane light rotation (only valid in `lambertian` rendering mode).
* `W` to toggle wireframe rendering on top of current rendering mode.