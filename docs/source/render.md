# Mesh Renderer

We provide a convenient mesh renderer based on:
* `kiui.mesh.Mesh` 
* `nvdiffrast`
* `dearpygui`


### Usage
```bash
kire --help 
# this is short for
python -m kiui.render --help

# open a GUI to render a mesh (extra dep: nvdiffrast)
kire mesh.obj
kire mesh.glb --pbr # render with PBR (metallic + roughness)
kire mesh.obj --save_video out.mp4 --wogui # save 360 degree rotating video
```