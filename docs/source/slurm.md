# Slurm Utilities

[[source]](https://github.com/ashawkey/kiuikit/blob/main/kiui/slurm.py)

Wraps Slurm CLI with clean and beautiful output.

## Examples

```bash
python -m kiui.slurm --help
# short cut:
ks --help

# check cluster information (sinfo)
ks info
ks i # short cut

# check node details (scontrol show node)
ks i <nodename>

# check job details (scontrol show job)
ks job <jobid>
ks j <jobid> # short cut

# show current user's jobs (squeue)
ks queue
ks q # short cut

# show all users' jobs
ks q -a

# show specific user's jobs
ks q -u <username>

# check job history (sacct)
ks history
ks h # short cut

# check job history for the last 7 days (default is 3)
ks h -d 7

# check job logs (stdout, last 100 lines)
ks log <jobid>
ks l <jobid> # short cut

# check job stderr
ks l <jobid> -e

# check all job logs (no truncation)
ks l <jobid> -a

# check last N lines of job logs (default is 100)
ks l <jobid> -n 500

# interactive cancel jobs (will prompt for selection and confirmation)
ks cancel
ks c # short cut

# directly cancel specific job(s) by ID
ks c <jobid>
ks c <jobid1> <jobid2>

# show cluster resource usage by user (sorted by GPU count)
ks usage
ks u # short cut

# show top 10 users by resource usage
ks u -n 10

# filter usage by partition
ks u -p batch
```
