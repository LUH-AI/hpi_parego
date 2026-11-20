from __future__ import annotations

import copy
import json
import random
from pathlib import Path
import numpy as np
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Constant,
)
from ConfigSpace.util import deactivate_inactive_hyperparameters

from hpi_parego.hpi_utils import (
    GPImportanceModel,
    OracleModel,
    PseudoConstant,
    calculate_hpi_history,
    calculate_hpi_hypershap,
    random_selection,
    tunable_names,
)
from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.maximizer.abstract_acquisition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.model.gaussian_process.abstract_gaussian_process import AbstractGaussianProcess
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.logging import get_logger
from smac.utils.multi_objective import normalize_costs


logger = get_logger(__name__)


class MyRandomSearch(AbstractAcquisitionMaximizer):
    """Get candidate solutions via random sampling of configurations. Used for both multi-objective (ParEGO) and
    single-objective optimization."""
    def __init__(
        self,
        configspace: ConfigurationSpace=ConfigurationSpace(),
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        seed: int = 0,
        hpi_method='hypershap', # method for selecting the hyperparameters ('hypershap', 'random', 'mcts_vs')
        reduce_cs='incumbent',
        thresh=[0.0,0.8,0.8],
        n_trials=100,
        path_to_run=None,
        rnd_prob=0.1,
        hpi_surrogate='gp', # model explained by HyperSHAP ('gp' or 'model', the surrogate model of the optimizer)
        interactions=False,
        ground_truth=False,
        objective_function=None,
    ) -> None:
        super().__init__(
            configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            seed=seed,
        )
        print('PATH', path_to_run)
        assert hpi_method in ('hypershap', 'random', 'mcts_vs')
        assert hpi_surrogate in ('gp', 'model')
        assert not ground_truth or objective_function is not None, 'ground_truth needs objective_function'

        self.path_to_run = Path(path_to_run) if isinstance(path_to_run,str) else path_to_run
        self._original_cs = copy.deepcopy(configspace)
        self.hpi_method = hpi_method
        self.reduce_cs = reduce_cs # whether to reduce the configspace for sampling ('default', 'random', 'incumbent', 'no')
        self.thresh = thresh # threshold or threshlist for quantile cut off for hpi
        self.n_trials = n_trials
        self.incumbent = self._original_cs.sample_configuration()
        self.thresh_list = self.thresh if not isinstance(self.thresh, float) else None
        self.rnd_prob=rnd_prob
        self.seed = seed
        self._configspace.seed(self.seed)
        self._original_cs.seed(self.seed)
        self._rnd_branch_rng = random.Random(self.seed)
        self._hpi_rng = random.Random(self.seed)
        self.hpi_surrogate = hpi_surrogate
        self.interactions = interactions # Paper Section 4.4
        self.ground_truth = ground_truth # Paper Appendix Section J
        self.objective_function = objective_function # LazyObjective of the task, only built with ground_truth
        self.runhistory = None # set by the config selector
        self._gp = None # GP of the last step, refitted in the next one
        self._free_hps = {} # hpi_method='mcts_vs': tuned hyperparameters of every proposed configuration
        self._current_free = set(tunable_names(self._original_cs))

    def scalarize(self, costs):
        """Normalizes the costs with the runhistory's bounds and scalarizes them with the current ParEGO weights, as
        the surrogate model is trained on. Single-objective costs are returned as they are."""
        costs = np.atleast_2d(np.asarray(costs, dtype=float))
        if costs.shape[1] == 1:
            return costs[:, 0]
        theta, rho = np.asarray(self._acquisition_function._theta), self._acquisition_function._rho
        normed = np.array([normalize_costs(list(c), self.runhistory.objective_bounds) for c in costs])
        return (theta * normed).max(1) + rho * (theta * normed).sum(1)

    def observed_costs(self, configs):
        """Paper Section 3.2: observed costs of the configs, scalarized with the current weights (inf for crashed or
        unfinished trials), or None if no cost is available."""
        if self.runhistory is None:
            return None
        costs = []
        for c in configs:
            cost = np.atleast_1d(np.asarray(self.runhistory.average_cost(c), dtype=float))
            costs.append(self.scalarize(cost)[0] if np.all(np.isfinite(cost)) else np.inf)
        costs = np.array(costs)
        return costs if np.isfinite(costs).any() else None

    def importance_model(self, configs):
        """
        Paper Appendix Section D.1
        The model HyperSHAP explains: the ground truth (Appendix J), a GP fitted to the observed scalarized costs, or
        the surrogate model of the optimizer.
        """
        if self.ground_truth:
            return OracleModel(self.objective_function.get(), self._original_cs, self.scalarize)
        model = self._acquisition_function.model
        if self.hpi_surrogate == 'model' or isinstance(model, AbstractGaussianProcess):
            return model
        costs = self.observed_costs(configs)
        if costs is None:
            return model
        gp = GPImportanceModel(self._original_cs, convert_configurations_to_array(configs), costs, gp=self._gp,
                               seed=self.seed)
        self._gp = gp.gp
        return gp

    def mcts_vs_selection(self, previous_configs):
        """Paper Appendix Section D.2: MCTS-VS criterion; configurations proposed in the full space (initial design,
        initialization phase, random configurations) count as tuning all hyperparameters."""
        costs = self.observed_costs(previous_configs)
        if costs is None:
            return []
        everything = set(tunable_names(self._original_cs))
        free = [self._free_hps.get(self._config_key(c), everything) for c in previous_configs]
        return calculate_hpi_history(free, costs, self._original_cs, self.thresh, self._hpi_rng)[0]

    @staticmethod
    def _config_key(config):
        return tuple(sorted(dict(config).items()))

    def _free_in_configspace(self):
        """Names of the hyperparameters that are tuned in the current (possibly reduced) configspace."""
        return {hp.name for hp in self._configspace.values()
                if not isinstance(hp, (PseudoConstant, Constant)) and len(getattr(hp, 'choices', (0, 1))) > 1}

    def maximize(self, *args, **kwargs):
        # SMAC's random design samples from the (reduced) configspace, so its configurations are mapped back too
        for config in super().maximize(*args, **kwargs):
            if config.config_space is not self._original_cs:
                origin = config.origin
                config = deactivate_inactive_hyperparameters(dict(config), self._original_cs)
                config.origin = origin
            if self.hpi_method == 'mcts_vs':
                self._free_hps.setdefault(self._config_key(config), self._current_free)
            yield config

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
        _sorted: bool = True,
    ) -> list[tuple[float, Configuration]]:
        """
        Paper: Section 3.1
        Maximize acquisition function with random search

        This function performs an adaptive search over the configuration space,
        possibly reducing it based on HPI.

        It uses HyperSHAP to identify important hyperparameters
        and restricts the search space accordingly.

        Parameters
        ----------
        previous_configs : list[Configuration]
            Previously evaluated configurations.
        n_points : int
            Number of configurations to return.
        _sorted : bool, optional
            If True, sort candidates by their acquisition value (descending)

        Returns
        -------
        list[tuple[float, Configuration]]
            Candidates sorted by their acquisition function value. (acq value, candidate)
        """

        if self.reduce_cs=='incumbent': # Incumbent: observed configuration with the lowest scalarized cost
            costs = self.observed_costs(previous_configs)
            if costs is None:
                X = convert_configurations_to_array(previous_configs)
                costs = self._acquisition_function.model.predict_marginalized(X)[0]
            self.incumbent = previous_configs[int(np.argmin(costs))]

        if self.thresh_list: # Paper: Section 3.3
            rh_path = self.path_to_run / "runhistory.json"
            if rh_path.exists():
                with rh_path.open() as f:
                    current_trial = len(json.load(f)["data"])
            else:
                current_trial = 0
            pos = min(current_trial // (self.n_trials // len(self.thresh_list)), len(self.thresh_list) - 1)
            self.thresh = self.thresh_list[pos]
        logger.info("Calculating HPI and reducing configspace if necessary.")

        if self._rnd_branch_rng.random() < self.rnd_prob:
            rand_configs = self._original_cs.sample_configuration(n_points)
            rand_configs = [(0,cfg) for cfg in rand_configs if cfg not in previous_configs]
            logger.debug("Random configuration.")
            self._current_free = set(tunable_names(self._original_cs))
            return rand_configs
        if (self.thresh>0) and (self.reduce_cs!='no'):
            if self.reduce_cs=='default':
                ref_cfg = self._original_cs.get_default_configuration()
            elif self.reduce_cs=='incumbent':
                ref_cfg = self.incumbent
            else:
                #random
                ref_cfg = self._original_cs.sample_configuration()

            if self.hpi_method=='hypershap':
                important_hps, _ = calculate_hpi_hypershap(self.importance_model(previous_configs), self._original_cs,
                                                           self.thresh, ref_cfg, self._hpi_rng, self.interactions)
            elif self.hpi_method=='random':
                important_hps = random_selection(self._original_cs, self.thresh, self._hpi_rng)
            else:
                important_hps = self.mcts_vs_selection(previous_configs)

            if len(important_hps) > 0:
                self.reduce_configspace(important_hps)
            else:
                self._configspace = self._original_cs
        else:
            self._configspace = self._original_cs

        seen = set(previous_configs)
        try:
            if n_points > 1:
                rand_configs = self._configspace.sample_configuration(size=n_points)
            else:
                rand_configs = [self._configspace.sample_configuration()]

            # Constants of the reduced space are encoded differently, which would corrupt the surrogate model's
            # inputs, so the configurations are mapped back to the original space
            rand_configs = [deactivate_inactive_hyperparameters(dict(c), self._original_cs) for c in rand_configs]
            for i in range(len(rand_configs)):
                rand_configs[i].origin = "Acquisition Function Maximizer: Random Search (sorted)"
        except:
            rand_configs = []

        rand_configs = [c for c in rand_configs if c not in seen]
        i=0
        while len(rand_configs)<2:
            logger.info("No new configurations found. Falling back to original search space.")
            if n_points > 1:
                rand_configs = self._original_cs.sample_configuration(size=n_points)
            else:
                rand_configs = [self._original_cs.sample_configuration()]
            rand_configs = [c for c in rand_configs if c not in seen]
            i+=1
            if i>30:
                print('Could not find new config after 30 tries. Returning anyway.')
                break
        self._current_free = (self._free_in_configspace() if i == 0 else set(tunable_names(self._original_cs)))
        rand_configs = self._sort_by_acquisition_value(rand_configs)
        return rand_configs


    def reduce_configspace(self, important_hps):
        """
        Paper Section 3.2
        Reduce the configuration space by focusing on important hyperparameters.

        Unimportant hyperparameters are fixed toward a specific value
        (default, incumbent, or random), reducing the dimensionality of the search space.
        Unimportant conditional hyperparameters that are inactive in the incumbent are fixed to their default.

        Conditions are preserved if parent hyperparameters are kept.
        Parents are added to the important hyperparameters if children are important.

        Args:
            important_hps (_type_): list of important hyperpamaters
        """
        print('Reduce configspace')
        new_cs = ConfigurationSpace()
        new_cs.random.set_state(self._configspace.random.get_state())

        cond_li = []
        for cond in self._original_cs.conditions:
            if cond.child.name in important_hps:
                if cond.parent.name not in important_hps:
                    important_hps.append(cond.parent.name)
            if cond.parent.name in important_hps:
                cond_li.append(cond)

        print('imp_hp', important_hps)
        for hp in self._original_cs.values():
            if hp.name in important_hps:
                try:
                    new_cs.add(hp)
                except:
                    new_cs.add_hyperparameter(hp)
            else:
                if self.reduce_cs=='default':
                    hp_value = hp.default_value
                elif self.reduce_cs=='incumbent':
                    hp_value = self.incumbent[hp.name] if hp.name in self.incumbent else hp.default_value
                else:
                    hp_value = hp.sample_value()
                try:
                    new_hp = PseudoConstant(hp.name, hp_value)
                except:
                    new_hp = hp
                    print(f"Hyperparameter {hp} of type {type(hp)} not supported. Using old hp values.")

                try:
                    new_cs.add(new_hp)
                except:
                    new_cs.add_hyperparameter(new_hp)
        new_cs.add(cond_li)

        self._configspace = new_cs
