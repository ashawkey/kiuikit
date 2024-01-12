# Mesh Renderer

We provide a convenient mesh renderer based on:
* [kiui.mesh.Mesh](mesh.md): our mesh implementation.
* [nvdiffrast](https://github.com/NVlabs/nvdiffrast): an NVIDIA GPU is required, install with `pip install git+https://github.com/NVlabs/nvdiffrast`.
* [dearpygui](https://github.com/hoffstadt/DearPyGui): **a desktop is required (cannot be forwarded by ssh)**. For headless servers, you can still use `--wogui` for rendering images and videos.


### Usage
```bash
kire --help 
# this is short for
python -m kiui.render --help
```
```
usage: kire [-h] [--pbr] [--envmap ENVMAP] [--front_dir FRONT_DIR] [--mode {lambertian,albedo,normal,depth,pbr}]
            [--W W] [--H H] [--radius RADIUS] [--fovy FOVY] [--wogui] [--force_cuda_rast] [--save SAVE]
            [--elevation ELEVATION] [--num_azimuth NUM_AZIMUTH] [--save_video SAVE_VIDEO]
            mesh

positional arguments:
  mesh                  path to mesh (obj, ply, glb, ...)

optional arguments:
  -h, --help            show this help message and exit
  --pbr                 enable PBR material
  --envmap ENVMAP       hdr env map path for pbr
  --front_dir FRONT_DIR
                        mesh front-facing dir
  --mode {lambertian,albedo,normal,depth,pbr}
                        rendering mode
  --W W                 GUI width
  --H H                 GUI height
  --radius RADIUS       default GUI camera radius from center
  --fovy FOVY           default GUI camera fovy
  --wogui               disable all dpg GUI
  --force_cuda_rast     force to use RasterizeCudaContext.
  --save SAVE           path to save example rendered images
  --elevation ELEVATION
                        rendering elevation
  --num_azimuth NUM_AZIMUTH
                        number of images to render from different azimuths
  --save_video SAVE_VIDEO
                        path to save rendered video
```

### Examples
```bash
# open a GUI to render a mesh
kire mesh.obj
kire mesh.obj --force_cuda_rast # if you cannot use OpenGL backend (usually happens for headless servers), this will fallback to CUDA backend with some limitations (ref: https://nvlabs.github.io/nvdiffrast/#rasterizing-with-cuda-vs-opengl-new)

kire mesh.obj --front_dir +x # specify mesh front-facing dir, default is +z (OpenGL convention). You can use [+-][xyz] to specify axis, and [123] to specify clockwise rotation (per 90 degree).

kire mesh.glb --pbr # render with PBR (metallic + roughness)
kire mesh.glb --pbr --envmap env.hdr # specify hdr file

kire mesh.obj --save_video out.mp4 --wogui # save 360 degree rotating video
kire mesh.obj --save out --wogui # save rendered images to the out folder (controlled by --elevation and --num_azimuth)
```

In the GUI, you can use:
* `Left Drag` to rotate camera.
* `Middle Scroll` to scale camera.
* `Right Drag` to pan camera.
* `Space` to toggle rendering mode.
* `P` to toggle camera orbital rotation.
* `L` to toggle light rotation.