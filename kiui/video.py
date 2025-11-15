import os
import subprocess
import json
import math
from typing import Tuple, Sequence

import cv2
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from kiui.typing import *


def read_video(
    path: str,
    mode: Literal["float", "uint8", "torch", "tensor"] = "float",
    order: Literal["RGB", "BGR"] = "RGB",
) -> Tuple[Union[ndarray, Tensor], float]:
    """Read a video file into a tensor / numpy array.

    Args:
        path: Path to the video file.
        mode: Returned data type.
            - ``"uint8"``: uint8 numpy array, [T, H, W, 3], range [0, 255]
            - ``"float"``: float32 numpy array, [T, H, W, 3], range [0, 1]
            - ``"torch"`` / ``"tensor"``: float32 torch tensor, [T, H, W, 3], range [0, 1]
        order: Channel order, ``"RGB"`` or ``"BGR"``.

    Returns:
        video: Video frames in the requested format.
        fps: Frames per second of the video.
    """

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # OpenCV reads in BGR
        if order == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"no frames read from video: {path}")

    video = np.stack(frames, axis=0)  # [T, H, W, 3], uint8

    if mode == "uint8":
        return video, float(fps)
    elif mode == "float":
        return video.astype(np.float32) / 255.0, float(fps)
    elif mode in ["torch", "tensor"]:
        return torch.from_numpy(video.astype(np.float32) / 255.0), float(fps)
    else:
        raise ValueError(f"Unknown read_video mode {mode}")


def write_video(
    path: str,
    video: Union[Tensor, ndarray],
    fps: float,
    order: Literal["RGB", "BGR"] = "RGB",
    codec: str = "mp4v",
) -> None:
    """Write a video from frames.

    Args:
        path: Path to write the video file.
        video: Video frames, [T, H, W, C] where C is 3 or 4.
            Can be numpy array (uint8 or float in [0, 1]) or torch tensor.
        fps: Frames per second.
        order: Channel order of the input frames, ``"RGB"`` or ``"BGR"``.
        codec: FourCC codec string for OpenCV, e.g. ``"mp4v"``, ``"XVID"``.
    """

    if torch.is_tensor(video):
        video = video.detach().cpu().numpy()

    if video.ndim == 3:
        video = video[None, ...]  # [H, W, C] -> [1, H, W, C]
    if video.ndim != 4:
        raise ValueError(f"write_video expects [T, H, W, C], got shape {video.shape}")

    if video.dtype == np.float32 or video.dtype == np.float64:
        video = np.clip(video, 0.0, 1.0)
        video = (video * 255.0).astype(np.uint8)

    T, H, W, C = video.shape
    if C not in (3, 4):
        raise ValueError(f"write_video expects 3 or 4 channels, got {C}")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) != "" else None
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))

    for i in range(T):
        frame = video[i]
        if C == 4:
            frame = frame[..., :3]  # drop alpha
        if order == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()


def get_video_info(path: str) -> Dict[str, Any]:
    """Inspect a video file using ffprobe and return metadata.

    Requires ffmpeg / ffprobe to be installed in the system.

    Args:
        path: Path to the video file.

    Returns:
        dict with keys:
            - path
            - width, height
            - fps
            - duration (seconds)
            - codec
            - codec_tag (fourcc / sample entry, e.g. ``"hvc1"`` or ``"hev1"``)
            - bitrate (bits per second)
            - filesize (bytes)
            - num_frames
            - raw_size (uncompressed RGB size in bytes)
            - compression_ratio (raw_size / filesize)
    """

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,codec_tag_string,avg_frame_rate,nb_frames:format=duration,bit_rate",
        "-of",
        "json",
        path,
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)

    stream = data["streams"][0]
    fmt = data["format"]

    width = int(stream["width"])
    height = int(stream["height"])
    codec = stream.get("codec_name", "unknown")
    codec_tag = stream.get("codec_tag_string", None)

    fps_str = stream.get("avg_frame_rate", "0/0")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0

    duration = float(fmt.get("duration", 0.0))
    bitrate = int(fmt.get("bit_rate", 0))

    nb_frames_str = stream.get("nb_frames", None)
    if nb_frames_str is not None and nb_frames_str not in ("0", ""):
        num_frames = int(nb_frames_str)
    elif fps > 0 and duration > 0:
        num_frames = int(round(fps * duration))
    else:
        num_frames = 0

    filesize = os.path.getsize(path) if os.path.exists(path) else 0
    raw_size = width * height * 3 * num_frames  # assume 8-bit RGB
    compression_ratio = (raw_size / filesize) if filesize > 0 else 0.0

    return {
        "path": path,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "codec": codec,
        "codec_tag": codec_tag,
        "bitrate": bitrate,
        "filesize": filesize,
        "num_frames": num_frames,
        "raw_size": raw_size,
        "compression_ratio": compression_ratio,
    }


def print_video_info(path: str) -> None:
    """Pretty-print video information."""

    info = get_video_info(path)

    def _fmt_duration(seconds: float) -> str:
        if seconds <= 0:
            return "Unknown"
        m, s = divmod(int(round(seconds)), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d} ({seconds:.2f} s)"

    def _fmt_bytes(n: int) -> str:
        if n <= 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        idx = int(math.floor(math.log(n, 1024)))
        idx = max(0, min(idx, len(units) - 1))
        value = n / (1024 ** idx)
        if idx == 0:
            return f"{int(value)} {units[idx]}"
        return f"{value:.2f} {units[idx]}"

    console = Console()
    table = Table(title=f"Video Info: {info['path']}")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    codec = info["codec"]
    codec_tag = info.get("codec_tag", None) or "unknown"
    codec_hint = {
        "h264": "libx264 / h264_nvenc",
        "hevc": "libx265 / hevc_nvenc",
        "mpeg4": "mpeg4",
    }.get(codec, None)

    table.add_row("Resolution", f"{info['width']} x {info['height']}")
    table.add_row("FPS", f"{info['fps']:.3f}")
    table.add_row("Duration", _fmt_duration(info['duration']))
    if codec_hint is not None:
        table.add_row("Codec", f"{codec} (encoders: {codec_hint})")
    else:
        table.add_row("Codec", codec)
    table.add_row("Codec tag", codec_tag)
    table.add_row("Bitrate", f"{info['bitrate']} bps")
    table.add_row("File size", _fmt_bytes(info.get("filesize", 0)))

    cr = info.get("compression_ratio", 0.0)
    if cr > 0:
        table.add_row("Compression", f"{cr:.2f}x (raw / encoded)")
    else:
        table.add_row("Compression", "Unknown")

    console.print(table)


def resize_video(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    codec: Optional[str] = None,
    crf: Optional[int] = None,
    preset: str = "medium",
    fps: float = None,
) -> None:
    """Resize a video and save to a new file using ffmpeg.

    Args:
        input_path: Path to the input video.
        output_path: Path to the output video.
        width: Target width.
        height: Target height.
        codec: Video codec / encoder name for ffmpeg, e.g. ``"h264"``, ``"hevc"``,
            ``"libx264"``, ``"libx265"``, ``"mpeg4"``, ``"h264_nvenc"``, ``"hevc_nvenc"``.
            If None, try to pick a reasonable encoder based on the input codec.
        crf: Constant Rate Factor (quality, lower is better) for CRF-based codecs
            (e.g. libx264 / libx265). If None, the function will try to roughly
            match the source video's bitrate (scaled by resolution/fps) instead
            of using CRF. For ``"mpeg4"``, this is mapped to a quantizer value
            ``q:v`` internally when CRF is provided.
        preset: ffmpeg preset, e.g. ``"slow"``, ``"medium"``, ``"fast"``.
        fps: If not None, resample video to this FPS.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) != "" else None

    # inspect source
    info = get_video_info(input_path)
    src_w, src_h = info["width"], info["height"]
    src_fps = info["fps"] if info["fps"] > 0 else None
    src_bitrate = info["bitrate"]
    src_codec = (info["codec"] or "").lower()
    src_tag = (info.get("codec_tag") or "").lower()

    # choose codec if not specified
    if codec is None:
        if src_codec == "h264":
            codec = "libx264"
        elif src_codec in ["hevc", "h265"]:
            codec = "libx265"
        elif src_codec == "mpeg4":
            codec = "mpeg4"
        else:
            codec = "libx264"
    lower_codec = codec.lower()

    scale_filter = f"scale={width}:{height}:flags=lanczos"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        scale_filter,
        "-c:v",
        codec,
    ]

    # preserve container codec tag (sample entry) for better compatibility,
    # e.g., keep ``hvc1`` vs ``hev1`` for HEVC, or ``avc1`` vs ``avc3`` for H.264.
    if codec is None:
        pass
    else:
        # only apply tag if user did not explicitly override codec tag via other means
        if lower_codec in ["libx265", "hevc", "h265"] and src_tag in ["hvc1", "hev1"]:
            cmd += ["-tag:v", src_tag]
        elif lower_codec in ["libx264", "h264"] and src_tag in ["avc1", "avc3"]:
            cmd += ["-tag:v", src_tag]

    if fps is not None:
        cmd += ["-r", str(fps)]

    # quality / rate control
    if crf is None and src_bitrate > 0:
        # Match source bitrate scaled by resolution (and fps if changed)
        scale_ratio = (width * height) / max(1, src_w * src_h)
        if fps is not None and src_fps is not None and src_fps > 0:
            scale_ratio *= fps / src_fps

        target_bitrate = int(src_bitrate * scale_ratio)
        # clamp to a reasonable range
        target_bitrate = max(100_000, min(target_bitrate, src_bitrate * 2))

        kbps = max(1, target_bitrate // 1000)
        cmd += [
            "-b:v",
            f"{kbps}k",
            "-maxrate",
            f"{kbps}k",
            "-bufsize",
            f"{2 * kbps}k",
        ]
    else:
        # CRF-based mode (or bitrate unknown)
        if crf is None:
            crf = 18  # sensible default when we cannot infer bitrate

        if lower_codec == "mpeg4":
            # mpeg4 does not use CRF; use qscale instead (1 best, 31 worst)
            q = max(1, min(31, int(round(crf / 2))))  # map CRF 18 -> q≈9 as a heuristic
            cmd += [
                "-q:v",
                str(q),
            ]
        else:
            cmd += [
                "-crf",
                str(crf),
                "-preset",
                preset,
            ]

    cmd += [
        "-c:a",
        "copy",
        output_path,
    ]

    subprocess.run(cmd, check=True)


def split_video(
    input_path: str,
    output_dir: str,
    timestamps: Sequence[float],
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    uniform: float = None,
    drop_last: bool = False,
) -> None:
    """Split a long video into shorter clips given split timestamps.

    Args:
        input_path: Path to the input video.
        output_dir: Directory to save the clips.
        timestamps: Sequence of timestamps in seconds.
            - If ``drop_last=False`` and ``uniform=None``: treated as absolute
              boundaries; the first segment always starts at 0 and the last
              one ends at the video duration.
            - If ``drop_last=True`` and ``uniform is None``: treated as segment
              lengths; the first segment is [0, timestamps[0]], the second is
              [timestamps[0], timestamps[0] + timestamps[1]], etc. Any remaining
              tail of the video is dropped.
            Ignored if ``uniform`` is not None.
        codec: Video codec / encoder name for ffmpeg, e.g. ``"h264"``, ``"hevc"``, ``"libx264"``, ``"libx265"``, ``"mpeg4"``, ``"h264_nvenc"``, ``"hevc_nvenc"``.
        crf: Constant Rate Factor (quality, lower is better).
        preset: ffmpeg preset, e.g. ``"slow"``, ``"medium"``, ``"fast"``.
        uniform: If not None, split the video into uniform segments of this many
            seconds. If ``drop_last=True``, any remaining tail shorter than this
            interval is dropped. If ``drop_last=False``, a final shorter segment
            is kept. Cannot be used together with explicit ``timestamps``.
        drop_last: Drop any remaining part of the video that is not explicitly
            covered by timestamps or a full uniform interval.
    """

    os.makedirs(output_dir, exist_ok=True)

    info = get_video_info(input_path)
    duration = info["duration"]

    if uniform is not None and timestamps:
        raise ValueError("split_video: specify either timestamps or uniform, not both.")

    # build boundaries
    if uniform is not None:
        if uniform <= 0:
            raise ValueError("split_video: uniform interval must be > 0.")
        step = float(uniform)
        boundaries = [0.0]
        t = step
        while t < duration:
            boundaries.append(t)
            t += step
        if not drop_last and boundaries[-1] != duration:
            boundaries.append(duration)
    else:
        # timestamps-based splitting
        ts = [float(t) for t in timestamps if float(t) > 0.0]
        if drop_last:
            # interpret timestamps as segment lengths
            boundaries = [0.0]
            acc = 0.0
            for seg_len in ts:
                acc += seg_len
                if acc >= duration:
                    boundaries.append(duration)
                    break
                boundaries.append(acc)
        else:
            # interpret timestamps as absolute boundaries within (0, duration)
            ts = sorted(t for t in ts if t < duration)
            boundaries = [0.0] + ts
            if boundaries[-1] != duration:
                boundaries.append(duration)

    base = os.path.splitext(os.path.basename(input_path))[0]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        seg_len = max(0.0, end - start)
        seg_seconds = int(round(seg_len))

        out_path = os.path.join(output_dir, f"{base}_{i:03d}_{seg_seconds}.mp4")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-c:v",
            codec,
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-c:a",
            "copy",
            out_path,
        ]

        subprocess.run(cmd, check=True)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple video tools based on OpenCV + ffmpeg.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # info
    parser_info = subparsers.add_parser("info", help="Inspect video information.")
    parser_info.add_argument("path", type=str, help="Path to input video.")

    # resize
    parser_resize = subparsers.add_parser("resize", help="Resize a video and save to output.")
    parser_resize.add_argument("input", type=str, help="Path to input video.")
    parser_resize.add_argument("output", type=str, help="Path to output video.")
    parser_resize.add_argument("--width", type=int, required=True, help="Target width.")
    parser_resize.add_argument("--height", type=int, required=True, help="Target height.")
    parser_resize.add_argument(
        "--codec",
        type=str,
        default=None,
        choices=["h264", "hevc", "libx264", "libx265", "mpeg4", "h264_nvenc", "hevc_nvenc"],
        help="ffmpeg video codec / encoder name. If omitted, try to infer from the input.",
    )
    parser_resize.add_argument(
        "--crf",
        type=int,
        default=None,
        help="ffmpeg CRF (quality). If omitted, try to roughly match the input bitrate.",
    )
    parser_resize.add_argument("--preset", type=str, default="medium", help="ffmpeg preset.")

    # split
    parser_split = subparsers.add_parser("split", help="Split a video into clips.")
    parser_split.add_argument("input", type=str, help="Path to input video.")
    parser_split.add_argument("output_dir", type=str, help="Directory to save clips.")
    parser_split.add_argument(
        "--timestamps",
        type=float,
        nargs="*",
        default=None,
        help="Optional split timestamps in seconds, e.g. --timestamps 10 20 30.",
    )
    parser_split.add_argument(
        "--uniform",
        type=float,
        default=None,
        help="Uniform segment length in seconds, e.g. --uniform 10.",
    )
    parser_split.add_argument(
        "--drop_last",
        action="store_true",
        help="Drop any remaining tail not covered by timestamps or full uniform intervals.",
    )
    parser_split.add_argument(
        "--codec",
        type=str,
        default="libx264",
        choices=["h264", "hevc", "libx264", "libx265", "mpeg4", "h264_nvenc", "hevc_nvenc"],
        help="ffmpeg video codec / encoder name.",
    )
    parser_split.add_argument("--crf", type=int, default=18, help="ffmpeg CRF (quality).")
    parser_split.add_argument("--preset", type=str, default="medium", help="ffmpeg preset.")

    args = parser.parse_args()

    if args.cmd == "info":
        print_video_info(args.path)
    elif args.cmd == "resize":
        resize_video(
            input_path=args.input,
            output_path=args.output,
            width=args.width,
            height=args.height,
            codec=args.codec,
            crf=args.crf,
            preset=args.preset,
        )
    elif args.cmd == "split":
        # basic validation for CLI: require either timestamps or uniform
        if (not args.timestamps or len(args.timestamps) == 0) and args.uniform is None:
            raise ValueError("split: please provide either --timestamps or --uniform.")
        if args.uniform is not None and args.timestamps and len(args.timestamps) > 0:
            raise ValueError("split: please provide only one of --timestamps or --uniform, not both.")

        split_video(
            input_path=args.input,
            output_dir=args.output_dir,
            timestamps=args.timestamps or [],
            codec=args.codec,
            crf=args.crf,
            preset=args.preset,
            uniform=args.uniform,
            drop_last=args.drop_last,
        )


if __name__ == "__main__":
    main()
