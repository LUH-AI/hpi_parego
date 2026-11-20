# Starts all runs of Appendix I: qNParEGO with and without the HPI-based reduction on the selected LCBench tasks.
# Requires BoTorch (pip install -e .[qnparego]).

import subprocess
import time


def submit(commands):
    """Submits every command as a Slurm job via ../start_job.sh."""
    print("number of jobs", len(commands))
    for command in commands:
        if subprocess.run(command, shell=True).returncode != 0:
            print(f"Command failed: {command}")
        time.sleep(0.1)
    print("all jobs submitted")


optimizers = [
    'botorch=qnparego',
    'botorch=hpi_qnparego',
]
tasks = [
    'cfg_lcbench_7593',
    'cfg_lcbench_34539',
    'cfg_lcbench_126025',
    'cfg_lcbench_126026',
    'cfg_lcbench_126029',
    'cfg_lcbench_167104',
    'cfg_lcbench_167149',
    'cfg_lcbench_167168',
    'cfg_lcbench_167181',
    'cfg_lcbench_168329',
]
seeds = range(0, 10)

submit([
    f"sbatch ../start_job.sh '+task/task_configs/evaluation={task}' baserundir=results/ +optimizer/{optimizer} seed={seed}"
    for optimizer in optimizers for seed in seeds for task in tasks
])
