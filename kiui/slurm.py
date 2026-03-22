import argparse
import subprocess
import getpass
import re
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text
import questionary


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

class JobInfo:
    def __init__(self, jid: str, data: Dict[str, str], source: str):
        self.jid = jid
        self.data = data
        self.source = source # "scontrol" or "sacct"

    @property
    def stdout_path(self) -> Optional[str]:
        path = self.data.get("StdOut")
        if not path or path in ["(null)", "N/A", "None", "/dev/null"]:
            return None
        return self._resolve_path(path)

    @property
    def stderr_path(self) -> Optional[str]:
        path = self.data.get("StdErr")
        if not path or path in ["(null)", "N/A", "None", "/dev/null"]:
            return None
        return self._resolve_path(path)
    
    @property
    def user(self) -> Optional[str]:
        # scontrol: UserId=user(uid)
        uid_str = self.data.get("UserId")
        if uid_str:
            match = re.search(r"([^\s\(]+)", uid_str)
            if match:
                return match.group(1)
        # sacct: User
        return self.data.get("User")
    
    @property
    def job_name(self) -> Optional[str]:
        return self.data.get("JobName")

    @property
    def node_list(self) -> Optional[str]:
        val = self.data.get("NodeList")
        if not val or val in ["(null)", "N/A", "None"]:
            return None
        return val

    def _resolve_path(self, path: str) -> str:
        if not path:
            return path
        
        # Replace %j, %A with JobID
        path = path.replace("%j", self.jid)
        path = path.replace("%A", self.jid)
        
        # Replace %u with User
        u = self.user
        if u:
            path = path.replace("%u", u)
            
        # Replace %x with JobName
        x = self.job_name
        if x:
            path = path.replace("%x", x)
            
        return path

def get_job_info(target: str) -> List[JobInfo]:
    """
    Get job information for a job ID or Name.
    Returns a list of JobInfo objects (handles multiple jobs with same name).
    Tries `scontrol` first (active jobs), then `sacct` (history).
    """
    # 1. Resolve Job IDs
    job_ids = []
    if target.isdigit():
        job_ids.append(target)
    else:
        # Resolve name to IDs using squeue (active)
        cmd = ["squeue", "--name", target, "--noheader", "--format=%i"]
        code, out, err = _run_cmd(cmd)
        if code == 0 and out:
            job_ids = [line.strip() for line in out.splitlines() if line.strip()]
        
        # If not found in squeue, try finding recent jobs with this name in sacct
        if not job_ids:
            # Look back 7 days by default for name resolution
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            cmd = ["sacct", "-n", "-o", "JobID", "--name", target, "-S", start_date, "-X", "-P"]
            code, out, err = _run_cmd(cmd)
            if code == 0 and out:
                job_ids = [line.strip() for line in out.splitlines() if line.strip()]

    if not job_ids:
        console.print(f"[bold red]Error:[/bold red] No job found matching '{target}'.")
        return []

    results = []
    seen_ids = set()

    # Process each Job ID
    for jid in job_ids:
        if jid in seen_ids:
            continue
        seen_ids.add(jid)

        # A. Try scontrol (Active Jobs)
        cmd = ["scontrol", "show", "job", jid]
        code, out, err = _run_cmd(cmd)
        
        if code == 0:
            data = {}
            # Robust parsing for scontrol
            pattern = re.compile(r"\s+([A-Za-z0-9_]+)=")
            # Prepend space to match first key
            text_to_parse = " " + out.replace("\n", " ")
            
            matches = list(pattern.finditer(text_to_parse))
            for i in range(len(matches)):
                key = matches[i].group(1)
                start_val = matches[i].end()
                end_val = matches[i+1].start() if i + 1 < len(matches) else len(text_to_parse)
                val = text_to_parse[start_val:end_val].strip()
                data[key] = val
            
            results.append(JobInfo(jid, data, "scontrol"))
            continue

        # B. Try sacct (History)
        # Request specific fields we care about
        fields = "JobID,JobName%200,Partition,User,State,ExitCode,Start,End,Elapsed,NodeList,StdOut,StdErr"
        cmd_s = ["sacct", "-j", jid, "-o", fields, "-P", "--units=M"] # -P for pipe, --units=M to force MB
        code_s, out_s, err_s = _run_cmd(cmd_s)
        
        if code_s == 0 and out_s.strip():
            lines = out_s.splitlines()
            if len(lines) >= 2:
                headers = lines[0].split("|")
                # Use the first data row (allocation)
                values = lines[1].split("|")
                
                data = {}
                if len(values) == len(headers):
                    for h, v in zip(headers, values):
                        data[h] = v
                    
                    results.append(JobInfo(jid, data, "sacct"))
                    continue
    
    if not results:
         console.print(f"[bold red]Error:[/bold red] Could not retrieve info for job(s) {', '.join(job_ids)}")
         
    return results

def job_details(target: str):
    """
    Show detailed information for a job ID or job name.
    """
    infos = get_job_info(target)
    if not infos:
        return

    for info in infos:
        grid = Table.grid(expand=True)
        grid.add_column(style="bold cyan", max_width=25)
        grid.add_column()
        
        if info.source == "scontrol":
            keys = sorted(info.data.keys())
            for k in keys:
                grid.add_row(f"{k}:", info.data[k])
        else: # sacct
            for k, v in info.data.items():
                grid.add_row(f"{k}:", v)
            
        console.print(Panel(grid, title=f"Job Details ({info.source}): {info.jid}", border_style="blue"))


def node_details(target: str):
    """
    Show detailed information for a node using `scontrol show node` and `squeue`.
    """
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
    # Use %200j to prevent truncation
    cmd_q = ["squeue", "-w", target, "--noheader", "-o", "%i|%u|%200j|%t|%M"]
    code_q, out_q, err_q = _run_cmd(cmd_q)
    
    if code_q == 0 and out_q.strip():
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
        table.add_column("JobID")
        table.add_column("User", style="yellow")
        table.add_column("Name", overflow="fold")
        table.add_column("State")
        table.add_column("Time")
        
        for line in out_q.splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 5:
                table.add_row(*[p.strip() for p in parts])
        
        console.print(Panel(table, title=f"Jobs on {target}", border_style="green"))
    elif code_q == 0:
        console.print(Panel(f"No jobs running on {target}.", border_style="dim"))
    else:
        console.print(f"[red]Error fetching jobs: {err_q}[/red]")

def _get_unique_node_summary() -> Tuple[int, Dict[str, int]]:
    """Get unique node counts by state using sinfo -N (one line per node)."""
    cmd = ["sinfo", "-N", "-h", "-o", "%N|%T"]
    code, out, err = _run_cmd(cmd)

    node_state: Dict[str, str] = {}
    if code == 0 and out.strip():
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) < 2:
                continue
            node_name, state = parts[0].strip(), parts[1].strip()
            node_state[node_name] = state

    state_counts: Dict[str, int] = {"idle": 0, "mix": 0, "alloc": 0, "drain": 0, "down": 0, "other": 0}
    for state in node_state.values():
        s = state.replace("*", "").lower()
        if "idle" in s:
            state_counts["idle"] += 1
        elif "mix" in s:
            state_counts["mix"] += 1
        elif "alloc" in s:
            state_counts["alloc"] += 1
        elif "drain" in s:
            state_counts["drain"] += 1
        elif "down" in s:
            state_counts["down"] += 1
        else:
            state_counts["other"] += 1

    return len(node_state), state_counts


def info():
    """
    Wrapper for `sinfo` to show partition and node status.
    """
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

    for line in lines:
        parts = line.split("|")
        if len(parts) < 6:
            continue
        
        partition, avail, timelimit, nodes_str, state, nodelist = [p.strip() for p in parts]
        
        state_clean = state.replace("*", "")
        
        if "idle" in state_clean:
            state_style = "green"
        elif "mix" in state_clean:
            state_style = "yellow"
        elif "alloc" in state_clean:
            state_style = "blue"
        elif "drain" in state_clean:
            state_style = "red"
        elif "down" in state_clean:
            state_style = "bold red"
        else:
            state_style = "white"

        avail_style = "green" if avail == "up" else "red"

        table.add_row(
            partition,
            Text(avail, style=avail_style),
            timelimit,
            nodes_str,
            Text(state, style=state_style),
            nodelist
        )

    # Unique node summary (deduplicated across partitions)
    total_nodes, state_counts = _get_unique_node_summary()

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
    title = f"Slurm Nodes (Total: {total_nodes} | {summary_str})"

    console.print(Panel(table, title=title, border_style="blue"))


def log_output(target: Optional[str] = None, num_lines: Optional[int] = 100, show_all: bool = False, show_stderr: bool = False, user: Optional[str] = None):
    """
    Show logs for a job.
    """
    if target is None:
        if user is None:
            user = getpass.getuser()
        
        # Get running jobs only
        cmd = ["squeue", "-u", user, "-t", "RUNNING", "-o", "%i|%P|%200j|%u|%t|%M|%D", "--noheader"]
        code, out, err = _run_cmd(cmd)

        if code != 0:
            console.print(f"[bold red]Error fetching jobs:[/bold red] {err}")
            return

        if not out.strip():
            console.print(f"[green]No active jobs found for user {user}.[/green]")
            return

        choices = []
        lines = out.splitlines()
        for line in lines:
            parts = line.split("|")
            if len(parts) < 7:
                continue
            
            jobid, partition, name, username, state, time_str, nodes = [p.strip() for p in parts]
            
            # Format for display
            display_text = f"{jobid:<10} | {state:<10} | {partition:<10} | {name}"
            choices.append(questionary.Choice(display_text, value=jobid))

        if not choices:
            console.print(f"[green]No active jobs found for user {user}.[/green]")
            return

        target = questionary.select(
            "Select a job to view logs (Enter to confirm):",
            choices=choices,
            style=questionary.Style([
                ('qmark', 'fg:#673ab7 bold'),       
                ('question', 'bold'),               
                ('answer', 'fg:#f44336 bold'),      
                ('pointer', 'fg:#673ab7 bold'),     
                ('highlighted', 'fg:#673ab7 bold'), 
                ('selected', 'fg:#cc5454'),         
                ('separator', 'fg:#cc5454'),        
                ('instruction', ''),                
                ('text', ''),                       
                ('disabled', 'fg:#858585 italic')   
            ])
        ).ask()
        
        if not target:
            return

    infos = get_job_info(target)
    if not infos:
        return
        
    # Warn if multiple jobs found
    if len(infos) > 1:
        console.print(f"[yellow]Warning:[/yellow] Multiple jobs found for '{target}'. Showing logs for the first one: {infos[0].jid}")

    info = infos[0]
    
    stdout_path = info.stdout_path
    stderr_path = info.stderr_path

    if not stdout_path:
        console.print(f"[bold red]Error:[/bold red] StdOut path not found for job {info.jid}.")
        return

    files_to_read = [stdout_path]

    if show_stderr:
        if not stderr_path:
            console.print(f"[bold red]Error:[/bold red] StdErr path not found for job {info.jid}.")
            return
        if stderr_path == stdout_path:
            # still show stdout
            console.print(f"[bold yellow]Warning:[/bold yellow] StdOut and StdErr paths are the same for job {info.jid}.")
        else:
            # only show stderr
            files_to_read = [stderr_path]
        
    console.print(Panel(f"Reading logs for Job {info.jid}\nFiles: {', '.join(files_to_read)}", border_style="blue"))

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


def _cancel_jobs(job_ids: List[str], job_names: Optional[Dict[str, str]] = None):
    """Confirm and cancel a list of job IDs."""
    if not job_ids:
        console.print("[yellow]No jobs to cancel.[/yellow]")
        return

    console.print(f"[bold red]You are about to cancel {len(job_ids)} job(s):[/bold red]")
    for jid in job_ids:
        label = f"  - {jid}"
        if job_names and jid in job_names:
            label += f" ({job_names[jid]})"
        console.print(label)

    confirm = questionary.confirm("Are you sure?").ask()
    if not confirm:
        console.print("[yellow]Cancellation aborted.[/yellow]")
        return

    cmd_cancel = ["scancel"] + job_ids
    code, _, err = _run_cmd(cmd_cancel)
    if code == 0:
        console.print(f"[green]Successfully cancelled {len(job_ids)} job(s).[/green]")
    else:
        console.print(f"[bold red]Error cancelling jobs:[/bold red] {err}")


def cancel(targets: Optional[List[str]] = None, user: Optional[str] = None):
    """
    Cancel jobs. Direct mode if job IDs given, interactive otherwise.
    """
    if targets:
        _cancel_jobs(targets)
        return

    if user is None:
        user = getpass.getuser()

    cmd = ["squeue", "-u", user, "-o", "%i|%P|%200j|%u|%t|%M|%D", "--noheader"]
    code, out, err = _run_cmd(cmd)

    if code != 0:
        console.print(f"[bold red]Error fetching jobs:[/bold red] {err}")
        return

    if not out.strip():
        console.print(f"[green]No active jobs found for user {user}.[/green]")
        return

    choices = []
    job_map: Dict[str, str] = {}

    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 7:
            continue
        jobid, partition, name, username, state, time_str, nodes = [p.strip() for p in parts]
        display_text = f"{jobid:<10} | {state:<10} | {partition:<10} | {name}"
        choices.append(questionary.Choice(display_text, value=jobid))
        job_map[jobid] = name

    if not choices:
        console.print(f"[green]No active jobs found for user {user}.[/green]")
        return

    selected_job_ids = questionary.checkbox(
        "Select jobs to cancel (Enter to confirm):",
        choices=choices,
        style=questionary.Style([
            ('qmark', 'fg:#673ab7 bold'),
            ('question', 'bold'),
            ('answer', 'fg:#f44336 bold'),
            ('pointer', 'fg:#673ab7 bold'),
            ('highlighted', 'fg:#673ab7 bold'),
            ('selected', 'fg:#cc5454'),
            ('separator', 'fg:#cc5454'),
            ('instruction', ''),
            ('text', ''),
            ('disabled', 'fg:#858585 italic')
        ])
    ).ask()

    if not selected_job_ids:
        console.print("[yellow]No jobs selected.[/yellow]")
        return

    _cancel_jobs(selected_job_ids, job_map)


def queue(user: Optional[str] = None, all_users: bool = False):
    """
    Wrapper for `squeue` to show jobs.
    """
    # %i: JobID, %P: Partition, %j: Name, %u: User, %t: State, %M: Time, %D: Nodes, %R: Reason/NodeList, %V: SubmitTime
    # Use %200j to prevent truncation
    cmd = ["squeue", "-o", "%i|%P|%200j|%u|%t|%M|%D|%R|%V"]
    
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
    table.add_column("Name", overflow="fold")
    table.add_column("User", style="yellow")
    table.add_column("State")
    table.add_column("Time")
    table.add_column("Nodes", justify="right")
    table.add_column("Reason / NodeList", style="dim")

    for line in lines:
        parts = line.split("|")
        if len(parts) < 9:
            continue
        
        jobid, partition, name, username, state, time_str, nodes, reason, submit_time_str = [p.strip() for p in parts]
        
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
            
            # Calculate waiting time
            try:
                # squeue output example: 2023-10-27T10:00:00
                submit_time = datetime.fromisoformat(submit_time_str)
                delta = datetime.now() - submit_time
                
                days = delta.days
                seconds = delta.seconds
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                sec = seconds % 60
                
                # Format: D-HH:MM:SS or HH:MM:SS matching slurm format
                if days > 0:
                    time_fmt = f"{days}-{hours:02d}:{minutes:02d}:{sec:02d}"
                else:
                    time_fmt = f"{hours:02d}:{minutes:02d}:{sec:02d}"
                    
                time_str = Text(time_fmt, style="yellow")
            except Exception:
                pass
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
    # -X: no steps
    # -o: output format
    # -P: parsable output (pipe separated)
    # Use JobName%200 to ensure full name
    cmd = ["sacct", "-X", "-P", "-o", "JobID,JobName%200,Partition,User,State,ExitCode,Start,End,Elapsed"]
    
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
    table.add_column("Name", overflow="fold")
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

def _parse_gpu_count(tres_per_node: str) -> int:
    """Extract GPU count from squeue %b (tres-per-node) field."""
    if not tres_per_node or tres_per_node in ["N/A", "(null)"]:
        return 0
    m = re.search(r"gpu[^,]*?:(\d+)", tres_per_node)
    return int(m.group(1)) if m else 0


def usage(partition: Optional[str] = None, top_n: Optional[int] = None):
    """
    Show cluster resource usage grouped by user, sorted by GPU count.
    """
    cmd = ["squeue", "--noheader", "-o", "%u|%t|%D|%C|%b"]
    if partition:
        cmd.extend(["-p", partition])

    code, out, err = _run_cmd(cmd)
    if code != 0:
        console.print(f"[bold red]Error running squeue:[/bold red] {err}")
        return

    users: Dict[str, Dict[str, int]] = {}

    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        user, state, nodes_str, cpus_str, tres_per_node = [p.strip() for p in parts]

        if user not in users:
            users[user] = {"running": 0, "pending": 0, "nodes": 0, "gpus": 0, "cpus": 0}

        if state == "R":
            try:
                n_nodes = int(nodes_str)
                n_cpus = int(cpus_str)
            except ValueError:
                n_nodes = 0
                n_cpus = 0
            gpus_per_node = _parse_gpu_count(tres_per_node)
            users[user]["running"] += 1
            users[user]["nodes"] += n_nodes
            users[user]["cpus"] += n_cpus
            users[user]["gpus"] += gpus_per_node * n_nodes
        elif state == "PD":
            users[user]["pending"] += 1

    if not users:
        msg = "No jobs found."
        if partition:
            msg = f"No jobs found in partition '{partition}'."
        console.print(f"[green]{msg}[/green]")
        return

    sorted_users = sorted(users.items(), key=lambda x: (x[1]["gpus"], x[1]["nodes"]), reverse=True)

    if top_n is not None and top_n > 0:
        sorted_users = sorted_users[:top_n]

    totals = {"running": 0, "pending": 0, "nodes": 0, "gpus": 0, "cpus": 0}
    for _, d in users.items():
        for k in totals:
            totals[k] += d[k]

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan", expand=True)
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("User", style="bold yellow")
    table.add_column("Running", justify="right")
    table.add_column("GPUs", justify="right", style="bold")
    table.add_column("Nodes", justify="right")
    table.add_column("CPUs", justify="right")
    table.add_column("Pending", justify="right", style="dim")

    for rank, (user, d) in enumerate(sorted_users, 1):
        table.add_row(
            str(rank),
            user,
            str(d["running"]),
            str(d["gpus"]),
            str(d["nodes"]),
            str(d["cpus"]),
            str(d["pending"]),
        )

    title_parts = [f"Running: {totals['running']} jobs",
                   f"{totals['gpus']} GPUs",
                   f"{totals['nodes']} nodes"]
    if totals["pending"] > 0:
        title_parts.append(f"Pending: {totals['pending']}")
    subtitle = f"{len(users)} users"
    if partition:
        subtitle += f" | partition: {partition}"
    if top_n is not None and top_n < len(users):
        subtitle += f" | showing top {top_n}"
    title = f"Cluster Usage ({' | '.join(title_parts)})"

    console.print(Panel(table, title=title, subtitle=subtitle, border_style="blue"))


def main():
    parser = argparse.ArgumentParser(prog="ks", description="kiui slurm utilities")
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
    l_parser.add_argument("target", nargs="?", help="Job ID or Name")
    l_parser.add_argument("--lines", "-n", type=int, default=100, help="Output the last N lines (default: 100)")
    l_parser.add_argument("--all", "-a", action="store_true", help="Output all lines")
    l_parser.add_argument("--stderr", "-e", action="store_true", help="Include stderr output")
    l_parser.add_argument("--user", "-u", type=str, help="Show jobs from specific user (interactive mode only)")

    # Cancel command
    c_parser = sub.add_parser("cancel", aliases=["c"], help="Cancel jobs (interactive if no IDs given)")
    c_parser.add_argument("targets", nargs="*", help="Job ID(s) to cancel directly (interactive if omitted)")
    c_parser.add_argument("--user", "-u", type=str, help="Show jobs from specific user (interactive mode only)")

    # Usage command
    u_parser = sub.add_parser("usage", aliases=["u"], help="Show cluster resource usage by user")
    u_parser.add_argument("--top", "-n", type=int, default=None, help="Show only the top N users")
    u_parser.add_argument("--partition", "-p", type=str, default=None, help="Filter by partition")

    args = parser.parse_args()

    if args.cmd in ["info", "i"]:
        if args.target:
            node_details(args.target)
        else:
            info()
    elif args.cmd in ["job", "j"]:
        job_details(args.target)
    elif args.cmd in ["queue", "q"]:
        queue(user=args.user, all_users=args.all)
    elif args.cmd in ["history", "h"]:
        history(user=args.user, days=args.days, all_users=args.all)
    elif args.cmd in ["log", "l"]:
        log_output(args.target, num_lines=args.lines, show_all=args.all, show_stderr=args.stderr, user=args.user)
    elif args.cmd in ["cancel", "c"]:
        cancel(targets=args.targets or None, user=args.user)
    elif args.cmd in ["usage", "u"]:
        usage(partition=args.partition, top_n=args.top)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
