# Video

[[source]](https://github.com/ashawkey/kiuikit/blob/main/kiui/video.py)

Utilities for reading, writing, inspecting and processing videos. Also supports image inspection.

## CLI

You can use the video tools either via the module:

```bash
python -m kiui.video --help
```

or via the `kivi` shortcut:

```bash
kivi --help
```

### Inspect video (or image) information

```bash
kivi info input.mp4
```

This prints a rich table with:

- **Video**: Resolution, FPS, duration, frames, codec (with encoder hints), codec tag, bitrate, file size, compression ratio vs raw RGB.
- **Image**: Resolution, channels, dtype, format, file size, compression ratio (respects actual dtype for raw-size calculation — e.g. float32 EXRs).

Example output:

```
      Video Info: input.mp4      
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field       ┃ Value                                 ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Type        │ Video                                 │
│ Resolution  │ 640 x 480                             │
│ FPS         │ 30.000                                │
│ Duration    │ 00:00:05 (5.00 s)                     │
│ Frames      │ 150                                   │
│ Codec       │ h264 (encoders: libx264 / h264_nvenc) │
│ Codec tag   │ avc1                                  │
│ Bitrate     │ 1569577 bps                           │
│ File size   │ 957.99 KB                             │
│ Compression │ 140.92x (raw / encoded)               │
└─────────────┴───────────────────────────────────────┘
```

```bash
# Inspect an image
kivi info photo.jpg
```

### Resize a video

```bash
kivi resize input.mp4 output_640p.mp4 \
    --width 640 --height 480
```

By default, `kivi`:

- Infers a reasonable encoder (`libx264`, `libx265`, `mpeg4`, …) from the input codec.
- Tries to match the original bitrate per pixel (scaled by resolution / fps), to preserve quality and file size characteristics.
- Keeps the codec tag (e.g. `hvc1` vs `hev1`) when using HEVC / H.264 encoders for better compatibility.
- Warns and rounds the output size to the closest encoder-compatible dimensions when chroma subsampling requires it (for example, `libx265` / 4:2:0 output needs even width and height).
- Runs ffmpeg quietly on successful resizes, while still showing `kivi` warnings and ffmpeg errors.

You can also override the encoder, quality, and framerate:

```bash
kivi resize input.mp4 output.mp4 \
    --width 640 --height 480 \
    --codec libx264 \
    --crf 23 \
    --fps 30
```

The `--fps` flag resamples frames: for downsampling it drops frames; for upsampling it uses motion-compensated interpolation (`minterpolate`) for smoother motion.

### Create a share-friendly preview

```bash
# Create a preview that stays under 10 MB (default)
kivi preview input.mp4

# Specify output path and target size
kivi preview input.mp4 output.mp4 --target-mb 20

# Cap resolution and framerate
kivi preview input.mp4 --max-resolution 1280 --max-fps 24

# Adjust quality floor
kivi preview input.mp4 --crf 18 --preset slow

# Drop audio
kivi preview input.mp4 --audio-kbps 0
```

Preview uses **capped CRF** encoding: quality-driven CRF with a `maxrate` cap derived from the size target, so short/simple clips keep high quality while long/large ones stay under the cap. It defaults to H.264 / yuv420p / mp4 with `faststart` and AAC audio for maximum compatibility.

When the input is too large for the budget, the function downsamples — preferring **resolution reduction over framerate reduction** — until the target bits-per-pixel is met.

Full options:

| Option | Default | Description |
|--------|---------|-------------|
| `--target-mb` | 10.0 | Target maximum file size in MB |
| `--codec` | libx264 | Video encoder (libx264, libx265, h264_nvenc, hevc_nvenc) |
| `--crf` | 23 | CRF quality ceiling (lower = better quality) |
| `--preset` | medium | ffmpeg preset (slow, medium, fast, etc.) |
| `--max-resolution` | 1920 | Cap on the longest spatial side (px) |
| `--min-resolution` | 480 | Floor on the longest side (px) |
| `--max-fps` | 30.0 | Cap on framerate |
| `--min-fps` | 15.0 | Floor on framerate |
| `--audio-kbps` | 128 | AAC audio bitrate; 0 to drop audio |

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

With `--drop_last`, timestamps are interpreted as **segment lengths** (not absolute boundaries): the first segment is `[0, 60]`, and any remaining tail is dropped.

Output clips are named as:

```text
<basename>_<idx>_<seconds>.mp4
```

where `<idx>` is a zero-padded index and `<seconds>` is the rounded duration of that segment.

You can also specify codec, CRF, and preset:

```bash
kivi split input.mp4 out_dir --uniform 10 \
    --codec libx265 --crf 22 --preset fast
```

## API

.. automodule:: kiui.video
   :members:
