# Slurm Utilities

[[source]](https://github.com/ashawkey/kiuikit/blob/main/kiui/slurm.py)

Wraps Slurm CLI with clean and beautiful output.

## Examples

```bash
python -m kiui.slurm --help
# short cut:
kis --help

# check cluster information (sinfo)
kis info
kis i # short cut

# check node details (scontrol show node)
kis i <nodename>

# check job details (scontrol show job)
kis job <jobid>
kis j <jobid> # short cut

# show current user's jobs (squeue)
kis queue
kis q # short cut

# show all users' jobs
kis q -a

# show specific user's jobs
kis q -u <username>
```