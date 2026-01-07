# Slurm Utilities

[[source]](https://github.com/ashawkey/kiuikit/blob/main/kiui/slurm.py)

Wraps Slurm CLI with clean and beautiful output.

## Examples

```bash
python -m kiui.slurm --help
# short cut:
kism --help

# check cluster information (sinfo)
kism info
kism i # short cut

# check node details (scontrol show node)
kism i <nodename>

# check job details (scontrol show job)
kism job <jobid>
kism j <jobid> # short cut

# show current user's jobs (squeue)
kism queue
kism q # short cut

# show all users' jobs
kism q -a

# show specific user's jobs
kism q -u <username>

# check job history (sacct)
kism history
kism h # short cut

# check job history for the last 7 days (default is 3)
kism h -d 7
```