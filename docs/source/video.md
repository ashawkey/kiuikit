# Video

[[source]](https://github.com/ashawkey/kiuikit/blob/main/kiui/video.py)

Utilities for reading, writing, inspecting and processing videos.

## CLI

You can use the video tools either via the module:

```bash
python -m kiui.video --help
```

or via the `kivi` shortcut:

```bash
kivi --help
```

### Inspect video information

```bash
kivi info input.mp4
```

This prints a rich table including:

- Resolution, FPS, duration
- Codec and codec tag (e.g., `hevc` + `hvc1` vs `hev1`)
- Bitrate, file size and estimated compression ratio vs raw RGB frames

### Resize a video

```bash
kivi resize input.mp4 output_640p.mp4 \
    --width 640 --height 480
```

By default, `kivi`:

- Infers a reasonable encoder (`libx264`, `libx265`, `mpeg4`, …) from the input codec.
- Tries to match the original bitrate per pixel (scaled by resolution / fps), to preserve quality and file size characteristics.
- Keeps the codec tag (e.g. `hvc1` vs `hev1`) when using HEVC / H.264 encoders for better compatibility.

You can also override the encoder and quality explicitly:

```bash
kivi resize input.mp4 output.mp4 \
    --width 640 --height 480 \
    --codec libx264 \
    --crf 23
```

### Split a video

```bash
# Split at custom timestamps (absolute seconds)
kivi split input.mp4 out_dir \
    --timestamps 10 20 30

# Split into uniform 10-second segments (last may be shorter)
kivi split input.mp4 out_dir \
    --uniform 10

# Only keep explicitly selected segments, drop the rest
kivi split input.mp4 out_dir \
    --timestamps 60 --drop_last
```

Output clips are named as:

```text
<basename>_<idx>_<seconds>.mp4
```

where `<idx>` is a zero-padded index and `<seconds>` is the rounded duration of that segment.


### API

.. automodule:: kiui.video
   :members: