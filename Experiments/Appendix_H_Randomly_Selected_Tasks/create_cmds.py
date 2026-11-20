# Starts all runs of Appendix H: HPI-ParEGO and ParEGO on 10 randomly selected LCBench and 10 randomly selected
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
    'smac20=multiobjective_rf +customoptimizer=parego',
    'smac20=multiobjective_rf +customoptimizer=hpi_parego',
]
tasks = [
    'cfg_lcbench_167184',
    'cfg_lcbench_167190',
    'cfg_lcbench_167200',
    'cfg_lcbench_167201',
    'cfg_lcbench_168335',
    'cfg_lcbench_168908',
    'cfg_lcbench_189354',
    'cfg_lcbench_189862',
    'cfg_lcbench_189865',
    'cfg_lcbench_3945',
    'cfg_rbv2_ranger_1040',
    'cfg_rbv2_ranger_14',
    'cfg_rbv2_ranger_1475',
    'cfg_rbv2_ranger_1493',
    'cfg_rbv2_ranger_18',
    'cfg_rbv2_ranger_24',
    'cfg_rbv2_ranger_29',
    'cfg_rbv2_ranger_41143',
    'cfg_rbv2_ranger_41164',
    'cfg_rbv2_ranger_41168',
]
seeds = range(0, 10)

submit([
    f"sbatch ../start_job.sh '+task/task_configs/random_selection={task}' baserundir=results/ +optimizer/{optimizer} seed={seed}"
    for optimizer in optimizers for seed in seeds for task in tasks
])
