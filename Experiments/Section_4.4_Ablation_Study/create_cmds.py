# Starts all runs of Section 4.4, Appendix D and Appendix J: HPI-ParEGO and variants that each change one component
# on the six ablation tasks, and two variants on the selected LCBench tasks (Appendix D.2 and D.6).

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


MAXIMIZER = "optimizer.smac_cfg.smac_kwargs.acquisition_maximizer"
SELECTOR = "optimizer.smac_cfg.smac_kwargs.config_selector"

# optimizer_id: overrides of the HPI-ParEGO defaults
variants = {
    "HPI-ParEGO": "",
    "hpi_random": f"{MAXIMIZER}.hpi_method=random",  # Appendix D.2
    "hpi_mcts_vs": f"{MAXIMIZER}.hpi_method=mcts_vs",  # Appendix D.2
    "fixing_default": f"{MAXIMIZER}.reduce_cs=default",  # Appendix D.3
    "fixing_random": f"{MAXIMIZER}.reduce_cs=random",  # Appendix D.3
    "tau_0-08-0": f"{MAXIMIZER}.thresh=[0.0,0.8,0.0]",  # Appendix D.4
    "tau_08-08-0": f"{MAXIMIZER}.thresh=[0.8,0.8,0.0]",  # Appendix D.4
    "tau_08": f"{MAXIMIZER}.thresh=[0.8]",  # Appendix D.4
    "r_0.0": f"{MAXIMIZER}.rnd_prob=0.0",  # Appendix D.5
    "r_0.2": f"{MAXIMIZER}.rnd_prob=0.2",  # Appendix D.5
    "u_6": f"{SELECTOR}.reweigh=6",  # Appendix D.6
    "u_20": f"{SELECTOR}.reweigh=20",  # Appendix D.6
    "interactions": f"{MAXIMIZER}.interactions=true",  # Appendix D.7
    "surrogate_rf": f"{MAXIMIZER}.hpi_surrogate=model",  # Appendix D.8
    "ground_truth": f"{MAXIMIZER}.ground_truth=true",  # Appendix J
}
ablation_tasks = [
    'cfg_lcbench_167161',
    'cfg_lcbench_167184',
    'cfg_lcbench_167190',
    'cfg_rbv2_ranger_469',
    'cfg_rbv2_ranger_40900',
    'cfg_rbv2_ranger_40981',
]

# Random selection (Appendix D.2) and u = 6 (Appendix D.6) on the selected LCBench tasks; HPI-ParEGO is taken from the
# runs of Section 4.5
lcbench_variants = ["hpi_random", "u_6"]
lcbench_tasks = [
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


def command(name, task_set, task, seed, results):
    return (f"sbatch ../start_job.sh '+task/task_configs/{task_set}={task}' baserundir={results}/ "
            f"+optimizer/smac20=multiobjective_rf +customoptimizer=hpi_parego seed={seed} optimizer_id={name} "
            f"{variants[name]}")


submit(
    [command(name, "ablation_study", task, seed, "results") for name in variants for seed in range(5)
     for task in ablation_tasks]
    + [command(name, "evaluation", task, seed, "results_lcbench") for name in lcbench_variants for seed in range(10)
       for task in lcbench_tasks]
)
