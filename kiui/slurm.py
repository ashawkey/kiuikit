import argparse
import subprocess
import sys
import getpass
import shutil
import time
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

console = Console()

def _run_cmd(cmd: Sequence[str], timeout: float = 30.0) -> Tuple[int, str, str]:
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

import re

def job_details(target: str):
    """
    Show detailed information for a job ID or job name using `scontrol show job`.
    """
    if not shutil.which("scontrol") or not shutil.which("squeue"):
        console.print("[bold red]Error:[/bold red] Slurm commands not found.")
        return

    # Try to determine if target is a Job ID or Name
    job_ids = []
    if target.isdigit():
        job_ids.append(target)
    else:
        # Assume it's a name, use squeue to find ID(s)
        # squeue --name <target> --noheader --format=%i
        cmd = ["squeue", "--name", target, "--noheader", "--format=%i"]
        code, out, err = _run_cmd(cmd)
        if code == 0 and out:
            job_ids = [line.strip() for line in out.splitlines() if line.strip()]
    
    if not job_ids:
        console.print(f"[bold red]Error:[/bold red] No job found matching '{target}'.")
        return

    for jid in job_ids:
        cmd = ["scontrol", "show", "job", jid]
        code, out, err = _run_cmd(cmd)
        
        if code != 0:
            console.print(f"[bold red]Error getting info for job {jid}:[/bold red] {err}")
            continue

        # Basic syntax highlighting for Key=Value
        highlighted_text = Text()
        pattern = re.compile(r"([A-Za-z0-9_]+)=")
        
        last_pos = 0
        for match in pattern.finditer(out):
            start, end = match.span()
            # Append text before match
            highlighted_text.append(out[last_pos:start])
            # Append styled key
            highlighted_text.append(out[start:end], style="bold cyan")
            last_pos = end
        
        # Append remaining text
        highlighted_text.append(out[last_pos:])
        
        console.print(Panel(highlighted_text, title=f"Job Details: {jid}", border_style="blue"))

def node_details(target: str):
    """
    Show detailed information for a node using `scontrol show node` and `squeue`.
    """
    if not shutil.which("scontrol") or not shutil.which("squeue"):
        console.print("[bold red]Error:[/bold red] Slurm commands not found.")
        return

    # 1. Get Node Info
    cmd = ["scontrol", "show", "node", target]
    code, out, err = _run_cmd(cmd)
    
    if code != 0:
        console.print(f"[bold red]Error getting info for node {target}:[/bold red] {err}")
        return

    # Extract key metrics
    metrics = {}
    for key in ["State", "CPUAlloc", "CPUTot", "RealMemory", "AllocMem", "Gres", "AllocTRES", "CfgTRES"]:
        match = re.search(rf"\b{key}=([^\s]+)", out)
        if match:
            metrics[key] = match.group(1)
        else:
            metrics[key] = "N/A"

    # Format Memory
    def to_gb(mb_str):
        try:
            val = float(mb_str)
            return f"{val / 1024:.1f} GB"
        except:
            return mb_str

    mem_str = f"{to_gb(metrics['AllocMem'])} / {to_gb(metrics['RealMemory'])}"
    cpu_str = f"{metrics['CPUAlloc']} / {metrics['CPUTot']}"
    
    # GPU parsing
    gpu_str = metrics['Gres']
    if gpu_str == "(null)" or gpu_str == "N/A":
        # Try to find gpu in CfgTRES
        if "gpu" in metrics['CfgTRES']:
             gpu_str = metrics['CfgTRES']

    # Create Summary Panel
    grid = Table.grid(expand=True)
    grid.add_column(style="bold cyan", max_width=20)
    grid.add_column()
    grid.add_row("State:", metrics['State'])
    grid.add_row("CPU (Alloc/Total):", cpu_str)
    grid.add_row("Memory (Alloc/Total):", mem_str)
    grid.add_row("Gres:", gpu_str)
    if metrics['AllocTRES'] != "N/A":
        grid.add_row("Alloc TRES:", metrics['AllocTRES'])

    console.print(Panel(grid, title=f"Node Details: [bold]{target}[/bold]", border_style="blue"))

    # 2. Get Running Jobs
    cmd_q = ["squeue", "-w", target, "--noheader", "-o", "%i|%u|%j|%t|%M"]
    code_q, out_q, err_q = _run_cmd(cmd_q)
    
    if code_q == 0 and out_q.strip():
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
        table.add_column("JobID")
        table.add_column("User", style="yellow")
        table.add_column("Name")
        table.add_column("State")
        table.add_column("Time")
        
        for line in out_q.splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 5:
                table.add_row(*parts)
        
        console.print(Panel(table, title=f"Jobs on {target}", border_style="green"))
    elif code_q == 0:
        console.print(Panel(f"No jobs running on {target}.", border_style="dim"))
    else:
        console.print(f"[red]Error fetching jobs: {err_q}[/red]")

def info():
    """
    Wrapper for `sinfo` to show partition and node status.
    """
    # Check if sinfo exists
    if not shutil.which("sinfo"):
        console.print("[bold red]Error:[/bold red] 'sinfo' command not found. Are you on a Slurm cluster?")
        return

    # Use pipe delimiter for easier parsing
    # %P: Partition, %a: Avail, %l: TimeLimit, %D: Nodes, %T: State, %N: NodeList
    cmd = ["sinfo", "-o", "%P|%a|%l|%D|%T|%N"]
    code, out, err = _run_cmd(cmd)
    
    if code != 0:
        console.print(f"[bold red]Error running sinfo:[/bold red] {err}")
        return

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Partition", style="bold")
    table.add_column("Avail")
    table.add_column("TimeLimit")
    table.add_column("Nodes", justify="right")
    table.add_column("State")
    table.add_column("NodeList", style="dim")

    lines = out.splitlines()
    if lines and "PARTITION" in lines[0]:
        lines = lines[1:]

    total_nodes_count = 0
    # Group states for summary
    state_counts = {"idle": 0, "mix": 0, "alloc": 0, "drain": 0, "down": 0, "other": 0}

    for line in lines:
        parts = line.split("|")
        if len(parts) < 6:
            continue
        
        partition, avail, timelimit, nodes_str, state, nodelist = [p.strip() for p in parts]
        
        # Colorize state
        state_clean = state.replace("*", "") # Remove * suffix if present
        
        if "idle" in state_clean:
            state_style = "green"
            count_key = "idle"
        elif "mix" in state_clean:
            state_style = "yellow"
            count_key = "mix"
        elif "alloc" in state_clean:
            state_style = "blue"
            count_key = "alloc"
        elif "drain" in state_clean:
            state_style = "red"
            count_key = "drain"
        elif "down" in state_clean:
            state_style = "bold red"
            count_key = "down"
        else:
            state_style = "white"
            count_key = "other"

        try:
            n_count = int(nodes_str)
            total_nodes_count += n_count
            if count_key in state_counts:
                state_counts[count_key] += n_count
            else:
                state_counts["other"] += n_count
        except ValueError:
            pass

        # Colorize avail
        avail_style = "green" if avail == "up" else "red"

        table.add_row(
            partition,
            Text(avail, style=avail_style),
            timelimit,
            nodes_str,
            Text(state, style=state_style),
            nodelist
        )

    # Summary string
    summary_parts = []
    if state_counts["idle"] > 0:
        summary_parts.append(f"[green]Idle: {state_counts['idle']}[/green]")
    if state_counts["mix"] > 0:
        summary_parts.append(f"[yellow]Mix: {state_counts['mix']}[/yellow]")
    if state_counts["alloc"] > 0:
        summary_parts.append(f"[blue]Alloc: {state_counts['alloc']}[/blue]")
    if state_counts["drain"] > 0:
        summary_parts.append(f"[red]Drain: {state_counts['drain']}[/red]")
    if state_counts["down"] > 0:
        summary_parts.append(f"[bold red]Down: {state_counts['down']}[/bold red]")
    
    summary_str = ", ".join(summary_parts)
    title = f"Slurm Nodes (Total: {total_nodes_count} | {summary_str})"

    console.print(Panel(table, title=title, border_style="blue"))


def log_output(target: str, num_lines: Optional[int] = 100, show_all: bool = False, show_stderr: bool = False):
    """
    Show logs for a job.
    """
    if not shutil.which("scontrol"):
        console.print("[bold red]Error:[/bold red] 'scontrol' command not found.")
        return

    # Resolve Job ID if name is provided
    job_ids = []
    if target.isdigit():
        job_ids.append(target)
    else:
        cmd = ["squeue", "--name", target, "--noheader", "--format=%i"]
        code, out, err = _run_cmd(cmd)
        if code == 0 and out:
            job_ids = [line.strip() for line in out.splitlines() if line.strip()]
    
    if not job_ids:
        console.print(f"[bold red]Error:[/bold red] No job found matching '{target}'.")
        return
        
    # Process the first job found
    jid = job_ids[0]
    if len(job_ids) > 1:
        console.print(f"[yellow]Warning:[/yellow] Multiple jobs found for '{target}'. Showing logs for the first one: {jid}")

    cmd = ["scontrol", "show", "job", jid]
    code, out, err = _run_cmd(cmd)
    
    if code != 0:
        console.print(f"[bold red]Error getting info for job {jid}:[/bold red] {err}")
        return

    # Parse StdOut and StdErr
    stdout_match = re.search(r"StdOut=([^\s]+)", out)
    stderr_match = re.search(r"StdErr=([^\s]+)", out)
    
    stdout_path = stdout_match.group(1) if stdout_match else None
    stderr_path = stderr_match.group(1) if stderr_match else None
    
    if not stdout_path or stdout_path == "(null)":
         console.print(f"[bold red]Error:[/bold red] StdOut path not found for job {jid}.")
         return

    files_to_read = [stdout_path]

    if show_stderr:
        if not stderr_path or stderr_path == "(null)":
            console.print(f"[bold red]Error:[/bold red] StdErr path not found for job {jid}.")
            return
        if stderr_path == stdout_path:
            # still show stdout
            console.print(f"[bold yellow]Warning:[/bold yellow] StdOut and StdErr paths are the same for job {jid}.")
        else:
            # only show stderr
            files_to_read = [stderr_path]
        
    console.print(Panel(f"Reading logs for Job {jid}\nFiles: {', '.join(files_to_read)}", border_style="blue"))

    # Construct command
    tail_cmd = ["tail"]
    
    if show_all:
        tail_cmd.extend(["-n", "+1"])
    elif num_lines is not None:
        tail_cmd.extend(["-n", str(num_lines)])
        
    tail_cmd.extend(files_to_read)
    
    try:
        # Use subprocess to stream output to stdout directly
        subprocess.run(tail_cmd, check=False)
    except KeyboardInterrupt:
        pass


def monitor(target: str):
    """
    Monitor GPU usage for a job using nvidia-smi via ssh.
    """
    if not shutil.which("scontrol") or not shutil.which("ssh"):
        console.print("[bold red]Error:[/bold red] scontrol or ssh not found.")
        return

    # Resolve Job ID
    job_ids = []
    if target.isdigit():
        job_ids.append(target)
    else:
        cmd = ["squeue", "--name", target, "--noheader", "--format=%i"]
        code, out, err = _run_cmd(cmd)
        if code == 0 and out:
            job_ids = [line.strip() for line in out.splitlines() if line.strip()]
    
    if not job_ids:
        console.print(f"[bold red]Error:[/bold red] No running job found matching '{target}'.")
        return
        
    jid = job_ids[0]
    if len(job_ids) > 1:
        console.print(f"[yellow]Warning:[/yellow] Multiple jobs found. Monitoring the first one: {jid}")

    # Get NodeList
    # Try squeue first as it's reliable for running jobs
    cmd = ["squeue", "-j", jid, "--noheader", "-o", "%N"]
    code, out, err = _run_cmd(cmd)
    
    nodelist_raw = None
    if code == 0 and out.strip():
        val = out.strip()
        if val != "N/A" and val != "(null)":
             nodelist_raw = val

    if not nodelist_raw:
        # Fallback to scontrol
        cmd = ["scontrol", "show", "job", jid]
        code, out, err = _run_cmd(cmd)
        if code != 0:
            console.print(f"[bold red]Error:[/bold red] {err}")
            return

        # Parse NodeList
        match = re.search(r"NodeList=([^\s]+)", out)
        if match:
             nodelist_raw = match.group(1)

    if not nodelist_raw or nodelist_raw == "(null)":
        console.print(f"[yellow]Job {jid} is not running on any nodes (Pending/Cancelled).[/yellow]")
        return

    # Expand NodeList
    cmd = ["scontrol", "show", "hostnames", nodelist_raw]
    code, out, err = _run_cmd(cmd)
    if code != 0:
        nodes = [nodelist_raw] # Fallback
    else:
        nodes = out.splitlines()

    try:
        console.print(f"[bold cyan]Monitoring Job {jid} on nodes: {', '.join(nodes)}[/bold cyan]")
        
        for node in nodes:
            console.print(Panel(f"Node: [bold]{node}[/bold]", border_style="blue"))
            # Run nvidia-smi
            ssh_cmd = ["ssh", node, "nvidia-smi"]
            subprocess.run(ssh_cmd) 

    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[bold red]Error reading logs:[/bold red] {e}")


def queue(user: Optional[str] = None, all_users: bool = False):
    """
    Wrapper for `squeue` to show jobs.
    """
    if not shutil.which("squeue"):
        console.print("[bold red]Error:[/bold red] 'squeue' command not found. Are you on a Slurm cluster?")
        return

    # %i: JobID, %P: Partition, %j: Name, %u: User, %t: State, %M: Time, %D: Nodes, %R: Reason/NodeList
    cmd = ["squeue", "-o", "%i|%P|%j|%u|%t|%M|%D|%R"]
    
    title = "All Jobs"
    if not all_users:
        if user is None:
            user = getpass.getuser()
        cmd.extend(["-u", user])
        title = f"Jobs for user: [bold gold1]{user}[/bold gold1]"
    
    code, out, err = _run_cmd(cmd)
    
    if code != 0:
        console.print(f"[bold red]Error running squeue:[/bold red] {err}")
        return

    lines = out.splitlines()
    if lines and "JOBID" in lines[0]:
        lines = lines[1:]
    
    if not lines:
        console.print(Panel(f"No jobs found for {user if user else 'all users'}.", title=title, border_style="green"))
        return

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan", expand=True)
    table.add_column("JobID", style="bold", no_wrap=True)
    table.add_column("Partition")
    table.add_column("Name")
    table.add_column("User", style="yellow")
    table.add_column("State")
    table.add_column("Time")
    table.add_column("Nodes", justify="right")
    table.add_column("Reason / NodeList", style="dim")

    for line in lines:
        parts = line.split("|")
        if len(parts) < 8:
            continue
        
        jobid, partition, name, username, state, time_str, nodes, reason = [p.strip() for p in parts]
        
        # Colorize state
        if state == "R": # Running
            state_style = "bold green"
            state_text = "RUNNING"
        elif state == "PD": # Pending
            state_style = "bold yellow"
            state_text = "PENDING"
        elif state == "CG": # Completing
            state_style = "bold blue"
            state_text = "COMPLETING"
        elif state == "CD": # Completed
            state_style = "green"
            state_text = "COMPLETED"
        elif state == "F": # Failed
            state_style = "red"
            state_text = "FAILED"
        elif state == "CA": # Cancelled
            state_style = "red"
            state_text = "CANCELLED"
        else:
            state_style = "white"
            state_text = state

        # Highligh reason if pending
        if state == "PD":
            reason_style = "yellow"
        else:
            reason_style = "dim"

        table.add_row(
            jobid,
            partition,
            name,
            username,
            Text(state_text, style=state_style),
            time_str,
            nodes,
            Text(reason, style=reason_style)
        )

    # Print the table directly
    console.print(Panel(table, title=title, border_style="green"))

def history(user: Optional[str] = None, days: int = 3, all_users: bool = False):
    """
    Wrapper for `sacct` to show job history.
    """
    if not shutil.which("sacct"):
        console.print("[bold red]Error:[/bold red] 'sacct' command not found. Are you on a Slurm cluster?")
        return

    # -X: no steps
    # -o: output format
    # -P: parsable output (pipe separated)
    cmd = ["sacct", "-X", "-P", "-o", "JobID,JobName,Partition,User,State,ExitCode,Start,End,Elapsed"]
    
    # Calculate start time
    start_date = datetime.now() - timedelta(days=days)
    cmd.extend(["-S", start_date.strftime("%Y-%m-%d")])

    title = f"Job History (Last {days} days)"
    if not all_users:
        if user is None:
            user = getpass.getuser()
        cmd.extend(["-u", user])
        title = f"Job History for user: [bold gold1]{user}[/bold gold1] (Last {days} days)"
    
    code, out, err = _run_cmd(cmd)
    
    if code != 0:
        console.print(f"[bold red]Error running sacct:[/bold red] {err}")
        return

    lines = out.splitlines()
    
    # Header check
    if not lines or "JobID" not in lines[0]:
        console.print(Panel(f"No job history found in the last {days} days.", title=title, border_style="blue"))
        return

    lines = lines[1:] # Skip header

    if not lines:
        console.print(Panel(f"No job history found in the last {days} days.", title=title, border_style="blue"))
        return

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan", expand=True)
    table.add_column("JobID", style="bold", no_wrap=True)
    table.add_column("Name")
    table.add_column("Partition")
    table.add_column("User", style="yellow")
    table.add_column("State")
    table.add_column("ExitCode")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Elapsed")

    for line in lines:
        parts = line.strip().split("|")
        # Ensure we have enough parts. 
        # JobID|JobName|Partition|User|State|ExitCode|Start|End|Elapsed
        if len(parts) < 9:
            continue
            
        jobid, name, partition, username, state, exitcode, start, end, elapsed = parts[:9]

        # Colorize state
        state_first = state.split()[0]
        
        if state_first == "COMPLETED":
            state_style = "green"
        elif state_first == "RUNNING":
            state_style = "bold green"
        elif state_first == "PENDING":
            state_style = "bold yellow"
        elif state_first in ["FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"]:
            state_style = "bold red"
        elif state_first.startswith("CANCELLED"):
            state_style = "red"
        else:
            state_style = "white"

        table.add_row(
            jobid,
            name,
            partition,
            username,
            Text(state, style=state_style),
            exitcode,
            start,
            end,
            elapsed
        )

    console.print(Panel(table, title=title, border_style="blue"))

def main():
    parser = argparse.ArgumentParser(prog="kis", description="kiui slurm utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Info command
    info_parser = sub.add_parser("info", aliases=["i"], help="Show cluster node and partition information (sinfo), or node details if name provided.")
    info_parser.add_argument("target", nargs="?", help="Node Name (optional). If provided, shows details for that node.")

    # Job command
    job_parser = sub.add_parser("job", aliases=["j"], help="Show detailed information for a job ID or job name using `scontrol show job`.")
    job_parser.add_argument("target", help="Job ID or Name.")

    # Queue command
    q_parser = sub.add_parser("queue", aliases=["q"], help="Show jobs (squeue)")
    q_parser.add_argument("--all", "-a", action="store_true", help="Show jobs from all users")
    q_parser.add_argument("--user", "-u", type=str, help="Show jobs from specific user")

    # History command
    h_parser = sub.add_parser("history", aliases=["h"], help="Show job history (sacct)")
    h_parser.add_argument("--days", "-d", type=int, default=3, help="Number of days to look back (default: 3)")
    h_parser.add_argument("--all", "-a", action="store_true", help="Show jobs from all users")
    h_parser.add_argument("--user", "-u", type=str, help="Show jobs from specific user")

    # Log command
    l_parser = sub.add_parser("log", aliases=["l"], help="Show job logs (stdout only by default)")
    l_parser.add_argument("target", help="Job ID or Name")
    l_parser.add_argument("--lines", "-n", type=int, default=100, help="Output the last N lines (default: 100)")
    l_parser.add_argument("--all", "-a", action="store_true", help="Output all lines")
    l_parser.add_argument("--stderr", "-e", action="store_true", help="Include stderr output")

    # Monitor command
    m_parser = sub.add_parser("monitor", aliases=["m"], help="Monitor GPU usage for a job")
    m_parser.add_argument("target", help="Job ID or Name")

    args = parser.parse_args()

    if args.cmd in ["info", "i"]:
        if args.target:
            node_details(args.target)
        else:
            info()
    elif args.cmd in ["job", "j"]:
        job_details(args.target)
    elif args.cmd in ["queue", "q"]:
        user = args.user
        queue(user=user, all_users=args.all)
    elif args.cmd in ["history", "h"]:
        user = args.user
        history(user=user, days=args.days, all_users=args.all)
    elif args.cmd in ["log", "l"]:
        log_output(args.target, num_lines=args.lines, show_all=args.all, show_stderr=args.stderr)
    elif args.cmd in ["monitor", "m"]:
        monitor(args.target)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

