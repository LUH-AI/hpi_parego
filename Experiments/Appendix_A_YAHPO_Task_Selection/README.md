# YAHPO Gym Task Subselection

This folder contains the code used to select YAHPO Gym benchmark instances (13 LCBench, 13 rbv2\_ranger) used in the paper (see Appendix A). HyperSHAP is run across all available datasets for each scenario; the results are then analyzed to pick the most representative tasks.

## Requirements

This submodule uses its own environment, separate from the main `hpi_parego` package (it requires **Python 3.12**).

1. Create and activate a Python environment:
   ```bash
   conda create -n yahpo_subselect python=3.12
   conda activate yahpo_subselect
   ```

2. Install dependencies from the `pyproject.toml` in this directory:
   ```bash
   pip install -e .
   ```

## Running the Experiment

The experiment is managed by [PyExperimenter](https://github.com/tornede/py_experimenter) and stores results in a local SQLite database.

1. On the first run, populate the experiment table from the config:
   ```python
   # In yahpo_hypershap.py, uncomment this line before running:
   pyexp.fill_table_from_config()
   ```

2. Run the experiment:
   ```bash
   python yahpo_hypershap.py
   ```
   Results are written to the SQLite database defined in `experiment_config_real.yaml` (`hpiparego.db`). The experiment covers two scenarios (`lcbench`, `rbv2_ranger`) across 20 datasets each, with 20 random weight vectors per dataset.

3. After the first run, comment `fill_table_from_config()` out again before re-running to avoid resetting the table.

## Evaluation

Open and run `select_yahpo_tasks.ipynb` to analyze the HyperSHAP results and reproduce the task selection.
