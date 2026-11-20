# Starts all runs of Appendix F: HPI-ParEGO and ParEGO on zdt2 with an increasing number of hyperparameters.

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
    'smac20=multiobjective_rf +customoptimizer=parego',
    'smac20=multiobjective_rf +customoptimizer=hpi_parego',
]
tasks = [
    'dim_2_zdt2',
    'dim_4_zdt2',
    'dim_6_zdt2',
    'dim_8_zdt2',
    'dim_10_zdt2',
    'dim_15_zdt2',
    'dim_20_zdt2',
    'dim_25_zdt2',
    'dim_30_zdt2',
]
seeds = range(0, 5)

submit([
    f"sbatch ../start_job.sh '+task/task_configs/PyMOO_scalability={task}' baserundir=results/ +optimizer/{optimizer} seed={seed}"
    for optimizer in optimizers for seed in seeds for task in tasks
])
