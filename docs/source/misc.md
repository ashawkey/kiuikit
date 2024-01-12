# Misc CLI

Misceallaneous command line interfaces.

### Usage

```python
# background removal utils
python -m kiui.cli.bg --help
python -m kiui.cli.bg input.png output.png
python -m kiui.cli.bg input_folder output_folder

# openpose detector
python -m kiui.cli.pose --help

# blip2 image captioning
python -m kiui.cli.blip --help

# hed edge detector
python -m kiui.cli.hed --help

# zoe depth estimation (extra dep: pip install timm==0.6.11)
python -m kiui.cli.depth_zoe --help

# midas depth estimation (dpt-large)
python -m kiui.cli.depth_midas --help

# sr (Real-ESRGAN from https://github.com/ai-forever/Real-ESRGAN/tree/main)
python -m kiui.sr --help
python -m kiui.sr image.jpg --scale 2 # save to image_2x.jpg
kisr image.jpg --scale 2 # short cut cmd

# made-in-heaven timer (https://github.com/ashawkey/made-in-heaven-timer)
python -m kiui.cli.timer --help

# mesh format conversion (only for a single textured mesh in obj/glb)
python -m kiui.cli.convert input.obj output.glb
kico input.obj output.glb # short cut cmd
kico mesh_folder/ video_folder --fmt .mp4 # render all meshes into rotating videos

# aesthetic predictor v2 (https://github.com/christophschuhmann/improved-aesthetic-predictor)
python -m kiui.cli.aes --help

# compare content of two directories
python -m kiui.cli.dircmp <dir1> <dir2>

# lock requirements.txt package versions based on current environment
python -m kiui.cli.lock_version <requirements.txt>
```