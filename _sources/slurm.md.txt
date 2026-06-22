# Slurm Utilities

[[source]](https://github.com/ashawkey/kiuikit/blob/main/kiui/slurm.py)

Wraps Slurm CLI with clean and beautiful output using `rich` tables and panels.

## Examples

```bash
python -m kiui.slurm --help
# short cut:
ks --help

# check cluster information (sinfo) — partitions, nodes, state with color
ks info
ks i # short cut

# check node details (scontrol show node) + running jobs on that node
ks i <nodename>

# check job details (scontrol show job / sacct fallback)
ks job <jobid>
ks j <jobid> # short cut
# also supports job name lookup (active jobs via squeue, then history via sacct)
ks j <jobname>

# show current user's jobs (squeue) with color-coded state, pending wait times
ks queue
ks q # short cut

# show all users' jobs
ks q -a

# show specific user's jobs
ks q -u <username>

# check job history (sacct), default last 3 days
ks history
ks h # short cut

# check job history for the last 7 days
ks h -d 7

# check job history for all users
ks h -a

# ── Log viewing ────────────────────────────────────────────────────

# Show per-rank training log (rank 0 by default, log_0.txt)
ks log <jobid>
ks l <jobid> # short cut

# Show a specific rank's log
ks l <jobid> -r 2

# Show the raw sbatch shell log (job.out)
ks l <jobid> -b

# Show the unified all-rank APS log
ks l <jobid> --aps

# With --batch, show stderr (job.err) instead
ks l <jobid> -b -e

# Show last 500 lines
ks l <jobid> -n 500

# Show all lines (no truncation)
ks l <jobid> -a

# Stream the log (tail -f)
ks l <jobid> -f

# Interactive: pick from your running jobs
ks l
ks l -u <username>

# ── Cancel jobs ────────────────────────────────────────────────────

# Interactive cancel (checkbox selection + confirmation)
ks cancel
ks c # short cut

# Direct cancel specific job(s) by ID
ks c <jobid>
ks c <jobid1> <jobid2>

# ── Pending jobs ───────────────────────────────────────────────────

# Show top 10 pending jobs sorted by priority (all users + your jobs)
ks pending
ks p # short cut

# Show top 20 pending jobs
ks p -n 20

# ── Resource usage ─────────────────────────────────────────────────

# Show cluster resource usage by user (sorted by GPU count)
ks usage
ks u # short cut

# Show top 10 users by resource usage
ks u -n 10

# Filter usage by partition
ks u -p batch
```
