import argparse
import subprocess
import sys
import getpass
import shutil
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

def main():
    parser = argparse.ArgumentParser(prog="kis", description="kiui slurm utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Info command
    info_parser = sub.add_parser("info", aliases=["i"], help="Show cluster node and partition information (sinfo), or job details if ID/Name provided.")
    info_parser.add_argument("target", nargs="?", help="Job ID or Name (optional). If provided, shows `scontrol show job` details.")

    # Queue command
    q_parser = sub.add_parser("queue", aliases=["q"], help="Show jobs (squeue)")
    q_parser.add_argument("--all", "-a", action="store_true", help="Show jobs from all users")
    q_parser.add_argument("--user", "-u", type=str, help="Show jobs from specific user")

    args = parser.parse_args()

    if args.cmd in ["info", "i"]:
        if args.target:
            job_details(args.target)
        else:
            info()
    elif args.cmd in ["queue", "q"]:
        user = args.user
        queue(user=user, all_users=args.all)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

