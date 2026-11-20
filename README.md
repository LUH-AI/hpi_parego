# Dynamic Hyperparameter Importance for Efficient Multi-Objective Optimization

This repository is the official implementation of ["Dynamic Hyperparameter Importance for Efficient Multi-Objective Optimization"](https://arxiv.org/abs/2601.03166).

## Requirements

The code requires **Python 3.10** and a Slurm-based cluster for running experiments.

1. Create and activate a Python environment:
   ```bash
   conda create -n hpi_parego python=3.10
   conda activate hpi_parego
   ```

2. Install [CARP-S](https://github.com/AutoML/CARP-S):
   ```bash
   git clone https://github.com/AutoML/CARP-S.git
   cd CARP-S
   pip install -e .
   ```

3. Install the required optimizers and benchmarks:
   ```bash
   python -m carps.build.make optimizer_smac optimizer_nevergrad optimizer_optuna benchmark_pymoo benchmark_yahpo
   ```

4. Patch the YAHPO installation (run from inside the `CARP-S` directory):
   ```bash
   python carps/container/recipes/benchmarks/YAHPO/patch_yahpo_configspace.py
   ```

5. Install the `hpi_parego` package (run from the `Code` directory containing `setup.py`):
   ```bash
   pip install -e .
   ```

6. Update the Hydra search path in `Experiments/start_job.sh` to point to the absolute path on your system:
   ```bash
   # Change this line:
   HYDRA_CMD="python -m carps.run hydra.searchpath=[file:////hpi_parego/configs]"
   # To, e.g.:
   HYDRA_CMD="python -m carps.run hydra.searchpath=[file:///absolute/path/to/Code/hpi_parego/configs]"
   ```

## Implementation

| File | Content |
| --- | --- |
| `hpi_parego/my_random_search.py` | Acquisition maximizer of HPI-ParEGO (Section 3): incumbent, HPI-based selection, reduction of the configuration space, τ-schedule. Also used for single-objective optimization. |
| `hpi_parego/hpi_utils.py` | HyperSHAP-based selection with a Gaussian process as importance model (Section 3.2, Appendix D.1) and the ablation variants: random selection and the MCTS-VS criterion (Appendix D.2), interactions (Appendix D.7), ground-truth importance (Appendix J). |
| `hpi_parego/my_config_selector.py` | SMAC's config selector, updating the ParEGO weights every `reweigh` evaluated configurations (u). |
| `hpi_parego/my_ei_hpi.py` | Expected improvement on the ParEGO scalarization. |
| `hpi_parego/qnparego.py` | qNParEGO with and without the HPI-based reduction (Appendix I). |
| `hpi_parego/igd_plus.py` | IGD+ on top of the CARP-S logs. |

The optimizer configs are in `hpi_parego/configs/`: `customoptimizer/hpi_parego.yaml` (HPI-ParEGO), `customoptimizer/parego.yaml` (ParEGO), `customoptimizer/hpi_soo.yaml` and `customoptimizer/soo.yaml` (single-objective optimization), `optimizer/botorch/` (qNParEGO, requires `pip install -e .[qnparego]`) and `optimizer/optuna/MO_EHVI.yaml` (EHVI, which is not in CARP-S). They are picked up automatically via the `hydra.searchpath` set in step 6 above.

## Running Experiments

1. Copy `Experiments/task_configs/` into `CARP-S/carps/configs/task/`. Each experiment folder corresponds to a part of the paper:

   | Experiment folder | Paper section | Task configs |
   | --- | --- | --- |
   | `Appendix_A_YAHPO_Task_Selection/` | Appendix A: selection of the YAHPO-Gym tasks (see the README in that folder) | – |
   | `Section_4.2_Single_Objective_Optimization/` | Section 4.2 | `SOO_evaluation/` |
   | `Section_4.3_HPI-ParEGO_on_PyMOO/` | Section 4.3, Appendix E | `PyMOO/` |
   | `Section_4.4_Ablation_Study/` | Section 4.4, Appendix D, Appendix J | `ablation_study/`, `evaluation/` |
   | `Section_4.5_HPI-ParEGO_on_HPO_Benchmarks/` | Section 4.5, Appendix G | `evaluation/` |
   | `Appendix_F_Scalability/` | Appendix F | `PyMOO_scalability/` |
   | `Appendix_H_Randomly_Selected_Tasks/` | Appendix H | `random_selection/` |
   | `Appendix_I_qNParEGO/` | Appendix I | `evaluation/` |

2. Navigate to the experiment folder under `Experiments/` and submit all jobs:
   ```bash
   python create_cmds.py
   ```
   Results are written to the `results/` folder next to the script. The ablation study additionally writes the runs on the selected LCBench tasks (Appendix D.2 and D.6) to `results_lcbench/` and uses HPI-ParEGO from the runs of Section 4.5 for them.

## Evaluation

After all jobs have finished, collect results and plot them:

1. Gather all run results of an experiment into a single file (in the experiment folder):
   ```bash
   python -m carps.analysis.gather_data results/
   ```
   For the ablation study, also run it for `results_lcbench/` and for the results of Section 4.5.

2. Open and run the `plotting.ipynb` notebook in the experiment folder to reproduce the figures from the paper. The figures are written to `figures/`. The plotting functions are shared across the experiments in `Experiments/plotting_utils.py`.

## Contributing

This code is released for reproducibility purposes. It is provided as-is and is not actively maintained.
