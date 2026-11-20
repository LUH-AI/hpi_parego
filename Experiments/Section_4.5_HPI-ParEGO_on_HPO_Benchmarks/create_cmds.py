# Starts all runs of Section 4.5 and Appendix G: HPI-ParEGO and the baselines on the selected LCBench and
# rbv2_ranger tasks.

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
    'cfg_rbv2_ranger_11',
    'cfg_rbv2_ranger_375',
    'cfg_rbv2_ranger_458',
    'cfg_rbv2_ranger_4135',
    'cfg_rbv2_ranger_4154',
    'cfg_rbv2_ranger_4534',
    'cfg_rbv2_ranger_4538',
    'cfg_rbv2_ranger_40496',
    'cfg_rbv2_ranger_41156',
    'cfg_rbv2_ranger_41157',
]
seeds = range(0, 10)

submit([
    f"sbatch ../start_job.sh '+task/task_configs/evaluation={task}' baserundir=results/ +optimizer/{optimizer} seed={seed}"
    for optimizer in optimizers for seed in seeds for task in tasks
])
