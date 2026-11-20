# Starts all runs of Section 4.3 and Appendix E: HPI-ParEGO and the baselines on the PyMOO ZDT tasks.

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
    'optuna=MO_TPE',
    'optuna=MO_NSGAII',
    'optuna=MO_EHVI',
    'nevergrad=DE',
    'smac20=multiobjective_rf +customoptimizer=parego',
    'smac20=multiobjective_rf +customoptimizer=hpi_parego',
]
tasks = [
    'zdt1',
    'zdt2',
    'zdt3',
    'zdt4',
    'zdt6',
]
seeds = range(0, 10)

submit([
    f"sbatch ../start_job.sh '+task/task_configs/PyMOO={task}' baserundir=results/ +optimizer/{optimizer} seed={seed}"
    for optimizer in optimizers for seed in seeds for task in tasks
])
