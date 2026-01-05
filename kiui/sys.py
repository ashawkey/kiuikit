import argparse
import datetime
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text



console = Console()

def _local_timezone_str() -> str:
    # Best-effort without root: use Python's tzinfo first, fall back to /etc/timezone if available.
    try:
        local_now = datetime.datetime.now().astimezone()
        tz = local_now.tzinfo
        tzname = tz.tzname(local_now) if tz else None
        offset = local_now.utcoffset()
        if offset is not None:
            total_min = int(offset.total_seconds() // 60)
            sign = "+" if total_min >= 0 else "-"
            total_min = abs(total_min)
            hh, mm = divmod(total_min, 60)
            off_str = f"UTC{sign}{hh:02d}:{mm:02d}"
        else:
            off_str = "UTC?"
        if tzname:
            return f"{tzname} ({off_str})"
        return off_str
    except Exception:
        pass
    tz_txt = (_read_text("/etc/timezone") or "").strip()
    return tz_txt or "-"


def _run_cmd(cmd: Sequence[str], timeout: float = 2.0) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", "not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def _kv_table(title: str, rows: Sequence[Tuple[str, Any]]) -> Table:
    # Intentionally no Table title: we usually wrap this inside a Panel(title=...),
    # and having both causes duplicated headings.
    t = Table(box=box.SIMPLE_HEAVY, show_header=False, expand=True)
    t.add_column("Key", style="bold cyan", no_wrap=True)
    t.add_column("Value", style="white")
    for k, v in rows:
        t.add_row(str(k), "-" if v is None else str(v))
    return t


def _bytes_fmt(n: Optional[float]) -> str:
    if n is None:
        return "-"
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    n = float(n)
    for u in units:
        if abs(n) < 1024.0:
            return f"{n:.2f} {u}"
        n /= 1024.0
    return f"{n:.2f} EiB"


def _percent(used: Optional[float], total: Optional[float]) -> str:
    if used is None or total in (None, 0):
        return "-"
    return f"{(100.0 * used / total):.1f}%"


def _parse_os_release() -> Dict[str, str]:
    txt = _read_text("/etc/os-release") or ""
    out: Dict[str, str] = {}
    for line in txt.splitlines():
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def _boot_time_utc() -> Optional[datetime.datetime]:
    # Linux: /proc/stat contains btime seconds since epoch.
    txt = _read_text("/proc/stat")
    if not txt:
        return None
    m = re.search(r"^btime\s+(\d+)\s*$", txt, flags=re.MULTILINE)
    if not m:
        return None
    try:
        return datetime.datetime.fromtimestamp(int(m.group(1)), tz=datetime.timezone.utc)
    except Exception:
        return None


def _uptime_seconds() -> Optional[float]:
    txt = _read_text("/proc/uptime")
    if not txt:
        return None
    try:
        return float(txt.split()[0])
    except Exception:
        return None


def _fmt_timedelta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = int(max(0, seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, sec = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m {sec:02d}s"
    return f"{hours:02d}h {minutes:02d}m {sec:02d}s"


def _cpu_model() -> Optional[str]:
    # Try reading from /proc/cpuinfo first (works on x86)
    txt = _read_text("/proc/cpuinfo")
    if txt:
        for line in txt.splitlines():
            if line.lower().startswith("model name"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()

    # Fallback: try `lscpu` (works on some ARM / other architectures where /proc/cpuinfo is lacking)
    code, out, _ = _run_cmd(["lscpu"], timeout=2.0)
    if code == 0 and out:
        for line in out.splitlines():
            if line.lower().startswith("model name"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()

    return None


def _mem_info_bytes() -> Dict[str, int]:
    # values are in kB.
    txt = _read_text("/proc/meminfo") or ""
    out: Dict[str, int] = {}
    for line in txt.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        m = re.search(r"(\d+)", v)
        if not m:
            continue
        out[k.strip()] = int(m.group(1)) * 1024
    return out


def _mounts() -> List[Tuple[str, str, str]]:
    # (device, mountpoint, fstype)
    txt = _read_text("/proc/mounts") or ""
    out: List[Tuple[str, str, str]] = []
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        dev, mnt, fstype = parts[0], parts[1], parts[2]
        out.append((dev, mnt, fstype))
    # de-dupe by mountpoint (keep first)
    seen = set()
    uniq: List[Tuple[str, str, str]] = []
    for dev, mnt, fstype in out:
        if mnt in seen:
            continue
        seen.add(mnt)
        uniq.append((dev, mnt, fstype))
    return uniq


def _disk_usage(mountpoint: str) -> Optional[shutil._ntuple_diskusage]:
    try:
        return shutil.disk_usage(mountpoint)
    except Exception:
        return None


def _gpu_info_nvidia_smi() -> List[Dict[str, str]]:
    # Non-root friendly. Prefer stable CSV query API.
    fields = [
        "index",
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "utilization.gpu",
        "temperature.gpu",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    code, out, _ = _run_cmd(cmd, timeout=2.5)
    if code != 0 or not out:
        return []
    rows: List[Dict[str, str]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        rows.append({fields[i]: parts[i] for i in range(len(fields))})
    return rows


def _gpu_info_fallback() -> str:
    # Basic visibility into GPU adapters if lspci exists.
    code, out, _ = _run_cmd(["lspci"], timeout=2.5)
    if code != 0 or not out:
        return "-"
    lines = [
        ln
        for ln in out.splitlines()
        if any(k in ln.lower() for k in ["vga compatible controller", "3d controller", "display controller"])
    ]
    return "\n".join(lines) if lines else "-"

def get_torch_info():
    rows: List[Tuple[str, Any]] = []

    # Python basics
    rows.append(("Python", sys.version.replace("\n", " ")))
    rows.append(("Executable", sys.executable))

    try:
        import torch  # type: ignore
    except Exception as e:
        console.print(Panel(_kv_table("Torch", [("torch", f"not installed ({type(e).__name__}: {e})")]), title="Torch Info"))
        return

    rows.append(("torch.__version__", getattr(torch, "__version__", "-")))

    # CUDA / cuDNN
    cuda_available = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
    rows.append(("torch.cuda.is_available()", cuda_available))
    rows.append(("torch.backends.cudnn.version()", getattr(torch.backends.cudnn, "version", lambda: None)()))

    # CUDA_HOME
    try:
        from torch.utils.cpp_extension import CUDA_HOME  # type: ignore
    except Exception:
        CUDA_HOME = None
    rows.append(("CUDA_HOME", CUDA_HOME))

    # Triton
    try:
        import triton  # type: ignore

        triton_ver = getattr(triton, "__version__", None)
        rows.append(("triton", f"installed ({triton_ver or 'unknown version'})"))
    except Exception as e:
        rows.append(("triton", f"not installed ({type(e).__name__})"))

    # nvcc (toolkit)
    code, out, err = _run_cmd(["nvcc", "--version"], timeout=2.5)
    if code == 0 and out:
        # show last ~2 lines (contains release & build)
        tail = "\n".join(out.splitlines()[-3:])
        rows.append(("nvcc --version", tail))
    else:
        rows.append(("nvcc --version", "-"))

    # NVIDIA driver (via nvidia-smi)
    gpus = _gpu_info_nvidia_smi()
    if gpus:
        # driver is per-machine; take from first row
        rows.append(("nvidia driver", gpus[0].get("driver_version", "-")))
    else:
        code, out, _ = _run_cmd(["nvidia-smi"], timeout=2.5)
        rows.append(("nvidia-smi", "available" if code == 0 else "not found"))

    # Per-device details from torch
    gpu_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    gpu_table.add_column("#", style="bold", no_wrap=True)
    gpu_table.add_column("Name", style="cyan")
    gpu_table.add_column("Capability", style="magenta", no_wrap=True)
    gpu_table.add_column("Total VRAM", justify="right", no_wrap=True)

    if cuda_available:
        try:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                cap = f"{props.major}.{props.minor}"
                total = _bytes_fmt(getattr(props, "total_memory", None))
                gpu_table.add_row(str(i), str(props.name), cap, total)
        except Exception:
            pass
    else:
        gpu_table.add_row("-", "CUDA not available", "-", "-")

    console.print(Panel(_kv_table("Torch", rows), title="Torch Info", border_style="green"))
    console.print(Panel(gpu_table, title="GPU Details", border_style="green"))


def get_os_info():
    # OS / platform
    u = platform.uname()
    osr = _parse_os_release()
    boot_utc = _boot_time_utc()
    uptime_s = _uptime_seconds()

    now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
    boot_local = boot_utc.astimezone() if boot_utc else None
    tz_str = _local_timezone_str()

    # CPU / load
    cpu_rows: List[Tuple[str, Any]] = []
    cpu_rows.append(("CPU count (logical)", os.cpu_count()))
    cpu_rows.append(("CPU model", _cpu_model()))
    try:
        la1, la5, la15 = os.getloadavg()
        cpu_rows.append(("Load avg (1/5/15m)", f"{la1:.2f} / {la5:.2f} / {la15:.2f}"))
    except Exception:
        cpu_rows.append(("Load avg (1/5/15m)", None))

    # Memory
    mem = _mem_info_bytes()
    mem_total = mem.get("MemTotal")
    mem_avail = mem.get("MemAvailable") or mem.get("MemFree")
    mem_used = (mem_total - mem_avail) if (mem_total is not None and mem_avail is not None) else None
    swap_total = mem.get("SwapTotal")
    swap_free = mem.get("SwapFree")
    swap_used = (swap_total - swap_free) if (swap_total is not None and swap_free is not None) else None

    mem_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    mem_table.add_column("Type", style="bold cyan", no_wrap=True)
    mem_table.add_column("Used", justify="right", no_wrap=True)
    mem_table.add_column("Total", justify="right", no_wrap=True)
    mem_table.add_column("Usage", justify="right", no_wrap=True)
    mem_table.add_row("RAM", _bytes_fmt(mem_used), _bytes_fmt(mem_total), _percent(mem_used, mem_total))
    mem_table.add_row("Swap", _bytes_fmt(swap_used), _bytes_fmt(swap_total), _percent(swap_used, swap_total))

    # Disks
    disk_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    disk_table.add_column("Mount", style="cyan")
    disk_table.add_column("FS", style="magenta", no_wrap=True)
    disk_table.add_column("Device", style="white")
    disk_table.add_column("Used", justify="right", no_wrap=True)
    disk_table.add_column("Total", justify="right", no_wrap=True)
    disk_table.add_column("Usage", justify="right", no_wrap=True)

    skip_fs = {
        "proc",
        "sysfs",
        "devtmpfs",
        "tmpfs",
        "devpts",
        "cgroup",
        "cgroup2",
        "pstore",
        "securityfs",
        "debugfs",
        "tracefs",
        "configfs",
        "overlay",
        "squashfs",
        "fusectl",
        "mqueue",
        "hugetlbfs",
        "rpc_pipefs",
        "autofs",
        "binfmt_misc",
        "nsfs",
    }

    # Collect candidates, then de-duplicate by underlying device.
    candidates: List[Tuple[str, str, str, float, float]] = []
    for dev, mnt, fstype in _mounts():
        if fstype in skip_fs:
            continue
        du = _disk_usage(mnt)
        if not du:
            continue
        # Exclude virtual mounts that report 0B total (not informative).
        if getattr(du, "total", 0) == 0:
            continue
        used = float(du.used)
        total = float(du.total)
        candidates.append((dev, mnt, fstype, used, total))

    # Best mount per device: prefer "/", else shortest mountpoint.
    best_by_dev: Dict[str, Tuple[str, str, float, float]] = {}
    for dev, mnt, fstype, used, total in candidates:
        prev = best_by_dev.get(dev)
        if prev is None:
            best_by_dev[dev] = (mnt, fstype, used, total)
            continue
        prev_mnt = prev[0]
        if prev_mnt == "/":
            continue
        if mnt == "/" or len(mnt) < len(prev_mnt):
            best_by_dev[dev] = (mnt, fstype, used, total)

    for dev in sorted(best_by_dev.keys(), key=lambda d: best_by_dev[d][0]):
        mnt, fstype, used, total = best_by_dev[dev]
        disk_table.add_row(mnt, fstype, dev, _bytes_fmt(used), _bytes_fmt(total), _percent(used, total))

    # GPU
    nvgpus = _gpu_info_nvidia_smi()
    gpu_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    gpu_table.add_column("#", style="bold", no_wrap=True)
    gpu_table.add_column("Name", style="cyan")
    gpu_table.add_column("Driver", style="magenta", no_wrap=True)
    gpu_table.add_column("VRAM Used/Total (MiB)", justify="right", no_wrap=True)
    gpu_table.add_column("Util %", justify="right", no_wrap=True)
    gpu_table.add_column("Temp C", justify="right", no_wrap=True)
    if nvgpus:
        for g in nvgpus:
            gpu_table.add_row(
                g.get("index", "-"),
                g.get("name", "-"),
                g.get("driver_version", "-"),
                f"{g.get('memory.used','-')}/{g.get('memory.total','-')}",
                g.get("utilization.gpu", "-"),
                g.get("temperature.gpu", "-"),
            )
    else:
        gpu_table.add_row("-", "No NVIDIA GPU info (nvidia-smi unavailable)", "-", "-", "-", "-")

    # Networking basics
    host_rows: List[Tuple[str, Any]] = []
    host_rows.append(("Hostname", socket.gethostname()))
    try:
        host_rows.append(("FQDN", socket.getfqdn()))
    except Exception:
        host_rows.append(("FQDN", None))

    # Prefer a "best effort" local IP without sending packets.
    ip = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = None
    host_rows.append(("Primary IPv4", ip))

    sys_rows: List[Tuple[str, Any]] = []
    sys_rows.append(("OS", osr.get("PRETTY_NAME") or platform.platform()))
    sys_rows.append(("Kernel", f"{u.system} {u.release} ({u.version})"))
    sys_rows.append(("Machine", f"{u.machine} / {u.processor or 'unknown'}"))
    sys_rows.append(("Timezone", tz_str))
    sys_rows.append(("Boot time", boot_local.isoformat(sep=" ", timespec="seconds") if boot_local else None))
    sys_rows.append(("Uptime", _fmt_timedelta(uptime_s)))
    sys_rows.append(("Now", now_utc.astimezone().isoformat(sep=" ", timespec="seconds")))

    # Build panels
    console.print(Panel(_kv_table("System", sys_rows), title="OS / System", border_style="blue"))
    console.print(Panel(_kv_table("CPU", cpu_rows), title="CPU", border_style="blue"))
    console.print(Panel(mem_table, title="Memory", border_style="blue"))
    console.print(Panel(disk_table, title="Storage", border_style="blue"))
    console.print(Panel(gpu_table, title="GPU", border_style="blue"))
    console.print(Panel(_kv_table("Network", host_rows), title="Network", border_style="blue"))


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(prog="kiss", description="kiui system utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("os", help="Print OS/system information")
    sub.add_parser("torch", help="Print torch/CUDA/triton information")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "os":
        get_os_info()
    elif args.cmd == "torch":
        get_torch_info()
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.cmd}")


if __name__ == "__main__":  # pragma: no cover
    main()