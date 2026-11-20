import json
import time

import numpy as np
from carps.objective_functions.yahpo import YAHPO_TASK_DATA_DIR, YahpoObjectiveFunction
from carps.utils.trials import TrialInfo
from ConfigSpace import Configuration, ConfigurationSpace, Constant
from hypershap import ExplanationTask, HyperSHAP
from py_experimenter.experimenter import PyExperimenter
from yahpo_gym import BenchmarkSet, local_config

local_config._config = {"data_path": str(YAHPO_TASK_DATA_DIR)}

class MultiObjectiveYAHPOGymBenchmark:
    """
    A benchmark class for multi-objective optimization using YAHPO Gym.

    This class enables evaluation of configurations on multiple objectives, normalizes their values,
    and aggregates them using user-defined weights. It also provides a reduced configuration space
    by excluding constant hyperparameters, except those explicitly specified.

    Attributes:
        scenario (str): The scenario name for the benchmark.
        dataset (str): The dataset identifier for the benchmark.
        metrics (list[str]): List of metric names to optimize.
        weights (list[float]): List of weights for each metric/objective.
        objective_functions (list[YahpoObjectiveFunction]): List of objective function instances.
        cs (ConfigurationSpace): The configuration space for the benchmark.
        default_config (dict): Default configuration values.
        add_values_of (list): List of hyperparameters to add default values for.
        mins (np.ndarray): Minimum observed values for each objective (for normalization).
        maxs (np.ndarray): Maximum observed values for each objective (for normalization).

    Methods:
        __init__(scenario, dataset, metrics, weights):
            Initializes the benchmark with scenario, dataset, metrics, and weights.

        evaluate(config):
            Evaluates a configuration by normalizing and aggregating objective values.

        get_config_space():
            Returns a reduced configuration space excluding constant hyperparameters,
            except those in `add_values_of`, and preserves all conditions.
    """

    def __init__(self, scenario:str, dataset:str, metrics:list[str], weights:list[float]) -> None:
        self.scenario = scenario
        self.dataset = dataset
        self.metrics = metrics
        self.weights = weights
        self.objective_functions = [YahpoObjectiveFunction(bench=scenario, instance=dataset, seed=42, metric=metric) for metric in self.metrics]
        self.cs = self.objective_functions[0].configspace
        self.default_config = dict(self.cs.get_default_configuration())
        self.add_values_of = []
        min_max_configs = self.cs.sample_configuration(size=1_000)
        obj_vals = np.array([[(-1) * obj.evaluate(TrialInfo(config=config)).cost for i, obj in enumerate(self.objective_functions)] for config in min_max_configs])
        self.mins = obj_vals.min(axis=0)
        self.maxs = obj_vals.max(axis=0)

    def evaluate(self, config: Configuration):
        """
        Evaluates a given configuration by normalizing and aggregating objective function values.

        Args:
            config (Configuration): The configuration to evaluate.

        Returns:
            float: The aggregated score computed as the weighted maximum of normalized objective values,
                   plus a small weighted sum term for diversity.
        """
        cfg_dict = dict(config)
        for x in self.add_values_of:
            cfg_dict[x] = self.default_config[x]
        config = Configuration(self.cs, cfg_dict)
        obj_vals = [((-1) * obj.evaluate(TrialInfo(config=config)).cost - self.mins[i])/(self.maxs[i]-self.mins[i]) for i, obj in enumerate(self.objective_functions)]
        return np.max(obj_vals*np.array(self.weights), axis=0) + 0.05 * np.sum(obj_vals * np.array(self.weights))

    def get_config_space(self) -> ConfigurationSpace:
        """
        Generates a reduced configuration space by excluding hyperparameters that are constants,
        unless they are listed in `self.add_values_of`. Non-constant hyperparameters are added
        to the reduced configuration space. All conditions from the original configuration space
        are preserved in the reduced space.

        Returns:
            ConfigurationSpace: The reduced configuration space containing only non-constant
            hyperparameters and all original conditions.
        """
        red_cs = ConfigurationSpace()

        for hp in self.cs:
            if isinstance(self.cs.get(hp), Constant):
                if hp not in self.add_values_of:
                    self.add_values_of.append(hp)
                continue

            red_cs.add(self.cs.get(hp))

        for cond in self.cs.conditions:
            red_cs.add(cond)

        print("Reduced ConfigSpace size: ", len(red_cs), " from ", len(self.cs))
        return red_cs


def run_experiment(config, result_processor, custom_config):
    """
    Runs a multi-objective hyperparameter importance experiment using the YAHPO Gym benchmark.

    Args:
        config (dict): Configuration dictionary containing keys:
            - "scenario" (str): The benchmark scenario name.
            - "dataset" (int or str): Index of the dataset to use.
            - "seed" (int or str): Random seed for reproducibility.
            - "num_iterations" (int or str): Number of experiment iterations.
        result_processor (object): Object responsible for processing and logging results.
        custom_config (dict): Additional custom configuration (currently unused).

    Raises:
        ValueError: If the scenario is not recognized.

    Side Effects:
        - Processes and logs metrics, hyperparameter importances, and computation time for each iteration.
        - Marks experiment completion via result_processor.

    Returns:
        None
    """
    scenario = config["scenario"]
    dataset_idx = int(config["dataset"])
    seed = int(config["seed"])
    num_iterations = int(config["num_iterations"])

    bench = BenchmarkSet(scenario)
    dataset = bench.instances[dataset_idx]

    if scenario == "lcbench":
        metrics = ["val_accuracy", "time"]
    elif scenario.startswith("rbv2_"):
        metrics = ["acc", "memory"]
    else:
        raise ValueError()

    result_processor.process_results({
        "metrics": json.dumps(metrics)
    })

    np.random.seed(seed)
    for i in range(num_iterations):
        weight_0 = np.random.rand()
        weights = [weight_0, 1-weight_0]

        objective_function = MultiObjectiveYAHPOGymBenchmark(scenario=scenario, dataset=dataset, metrics=metrics, weights=weights)
        hypershap = HyperSHAP(ExplanationTask.from_function(config_space=objective_function.get_config_space(), function=objective_function.evaluate, n_samples=10_000),n_workers=8)

        time_start = time.time()
        hypershap.tunability(objective_function.get_config_space().get_default_configuration(), index="SV", order=1)
        runtime = time.time() - time_start
        iv_w_names = list(hypershap.get_interaction_values_with_names().items())
        iv_w_names.sort(key=lambda i: i[1], reverse=True)
        log_iv = {}
        top_3 = []
        for item_key, item_value in iv_w_names:
            if len(item_key) > 0:
                log_iv[item_key[0]] = item_value
                if len(top_3) < 3:
                    top_3.append(item_key[0])
        top_3 = sorted(top_3)

        result_processor.process_logs({
            "hyperparameter_importances": {
                "weight_vector": json.dumps(weights),
                "hpi_res": json.dumps(log_iv),
                "top_3": json.dumps(top_3),
                "time_to_compute_hpi": str(runtime)
            }
        })

    result_processor.process_results({
        "done": "True"
    })
pyexp = PyExperimenter(experiment_configuration_file_path="experiment_config_real.yaml", database_credential_file_path="db_connection_data.yaml", use_codecarbon=False)
# pyexp.fill_table_from_config()
pyexp.execute(run_experiment, max_experiments=-1)