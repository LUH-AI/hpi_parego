from __future__ import annotations

import random

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from ConfigSpace import Configuration, ConfigurationSpace, Constant, UniformFloatHyperparameter, UniformIntegerHyperparameter
from gpytorch.mlls import SumMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from carps.optimizers.optimizer import Optimizer
from carps.utils.pareto_front import pareto
from carps.utils.trials import TrialInfo, TrialValue

from hpi_parego.hpi_utils import GPImportanceModel, calculate_hpi_hypershap, random_selection

RHO = 0.05
NUM_RESTARTS = 10
RAW_SAMPLES = 512
MC_SAMPLES = 128


class QNParEGOOptimizer(Optimizer):
    """
    Paper Appendix Section I
    BoTorch's qNParEGO on the unit cube of ConfigSpace's vector representation, optionally restricted to the
    important hyperparameters (all others fixed at the incumbent) as in hpi_parego.my_random_search.MyRandomSearch.

    Proposes batches of q candidates, handed out one per ask. A batch shares one weight vector, one selection and one
    random-branch draw, and is optimized jointly; the weights are redrawn every reweigh batches.
    """

    def __init__(
        self,
        task,
        loggers=None,
        expects_multiple_objectives: bool = True,
        expects_fidelities: bool = False,
        hpi_method: str = 'no', # 'no' (qNParEGO), 'hypershap' (HPI-qNParEGO) or 'random'
        thresh=(0.0, 0.8, 0.8),
        rnd_prob: float = 0.1,
        reweigh: int = 5,
        seed: int = 0,
        q: int = 2,
    ) -> None:
        super().__init__(task, loggers, expects_multiple_objectives=expects_multiple_objectives,
                         expects_fidelities=expects_fidelities)
        assert hpi_method in ('no', 'hypershap', 'random')
        self.configspace: ConfigurationSpace = self.task.objective_function.configspace
        hps = list(self.configspace.values())
        if self.configspace.conditions or self.configspace.forbidden_clauses:
            raise ValueError('qNParEGO supports only configuration spaces without conditions and forbiddens')
        unsupported = [hp.name for hp in hps
                       if not isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant))]
        if unsupported:
            raise ValueError(f'qNParEGO supports only uniform float/integer hyperparameters and constants, got {unsupported}')

        self.hpi_method = hpi_method
        self.thresh_list = list(thresh)
        self.rnd_prob = rnd_prob
        self.reweigh = reweigh
        self.seed = seed
        self.q = q
        self.n_trials = self.task.optimization_resources.n_trials

        self._hps = hps
        self.dims = [i for i, hp in enumerate(hps) if not isinstance(hp, Constant)]
        self.tunable = [hps[i].name for i in self.dims]
        self._template = self.configspace.get_default_configuration().get_array()

        self.configspace.seed(seed)
        torch.manual_seed(seed)
        self._rnd_branch_rng = random.Random(seed)
        self._hpi_rng = random.Random(seed)
        self._theta_rng = np.random.RandomState(seed)
        self._dup_rng = np.random.RandomState(seed)
        n_init = 2 * (len(self.dims) + 1)
        sobol = SobolEngine(len(self.dims), scramble=True, seed=seed).draw(n_init, dtype=torch.float64).numpy()
        self.initial_design = [self.vector_to_config(x) for x in sobol]

        self.history: list[tuple[TrialInfo, TrialValue]] = []
        self._batch: list[Configuration] = []
        self.theta: np.ndarray | None = None
        self._n_iter = 0
        self._gp = None
        self.thresh = self.thresh_list[0]
        self.incumbent: Configuration | None = None

    def vector_to_config(self, x: np.ndarray) -> Configuration:
        """Configuration of a point in the unit cube of the tunable dimensions. Integers are snapped by to_value, so
        config_to_vector of the result is the snapped point."""
        full = self._template.copy()
        full[self.dims] = x
        return Configuration(self.configspace, values={hp.name: hp.to_value(full[i]) for i, hp in enumerate(self._hps)})

    def sample_subspace(self, fixed_features: dict[int, float]) -> Configuration:
        x = self._dup_rng.random_sample(len(self.dims))
        for j, v in fixed_features.items():
            x[j] = v
        return self.vector_to_config(x)

    def config_to_vector(self, config: Configuration) -> np.ndarray:
        return config.get_array()[self.dims]

    def convert_configspace(self, configspace: ConfigurationSpace) -> ConfigurationSpace:
        return configspace

    def convert_to_trial(self, config: Configuration) -> TrialInfo:  # type: ignore[override]
        return TrialInfo(config=config)

    def _setup_optimizer(self) -> None:
        return None

    def scalarize(self, costs: np.ndarray) -> np.ndarray:
        """Augmented Chebyshev scalarization of costs, each objective min/max normalized over the finite observations
        (as SMAC's normalize_costs: an objective without range is set to 1). Non-finite rows get inf."""
        costs = np.asarray(costs, dtype=float)
        ok = np.isfinite(costs).all(1)
        lo, hi = costs[ok].min(0), costs[ok].max(0)
        span = hi - lo
        norm = np.where(span < 1e-10, 1.0, (costs - lo) / np.where(span < 1e-10, 1.0, span))
        t = self.theta * norm
        return np.where(ok, t.max(1) + RHO * t.sum(1), np.inf)

    def select(self, X_full: np.ndarray, scalarized: np.ndarray) -> list[str]:
        if self.hpi_method == 'random':
            return random_selection(self.configspace, self.thresh, self._hpi_rng)
        gp = GPImportanceModel(self.configspace, X_full, scalarized, gp=self._gp, seed=self.seed)
        self._gp = gp.gp
        return calculate_hpi_hypershap(gp, self.configspace, self.thresh, self.incumbent, self._hpi_rng)[0]

    def candidate(self, X: np.ndarray, costs: np.ndarray, fixed_features: dict[int, float]) -> np.ndarray:
        ok = np.isfinite(costs).all(1)
        train_x = torch.tensor(X[ok], dtype=torch.float64)
        train_y = -torch.tensor(costs[ok], dtype=torch.float64)
        d = len(self.dims)
        bounds = torch.stack([torch.zeros(d, dtype=torch.float64), torch.ones(d, dtype=torch.float64)])
        model = ModelListGP(*[
            SingleTaskGP(train_x, train_y[:, i:i + 1], input_transform=Normalize(d=d, bounds=bounds),
                         outcome_transform=Standardize(m=1))
            for i in range(train_y.shape[1])
        ])
        fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
        objective = GenericMCObjective(get_chebyshev_scalarization(
            torch.tensor(self.theta, dtype=torch.float64), train_y, alpha=RHO))
        acq = qLogNoisyExpectedImprovement(model=model, X_baseline=train_x, objective=objective,
                                           sampler=SobolQMCNormalSampler(torch.Size([MC_SAMPLES])),
                                           prune_baseline=True)
        cand, _ = optimize_acqf(acq, bounds=bounds, q=self.q, num_restarts=NUM_RESTARTS,
                                raw_samples=RAW_SAMPLES, fixed_features=fixed_features or None)
        return cand.detach().numpy()

    def ask(self) -> TrialInfo:
        n = len(self.history)
        if n < len(self.initial_design):
            return self.convert_to_trial(self.initial_design[n])
        if not self._batch:
            self._batch = self.next_batch()
        return self.convert_to_trial(self._batch.pop(0))

    def next_batch(self) -> list[Configuration]:
        n = len(self.history)

        if self._n_iter % self.reweigh == 0:
            theta = self._theta_rng.rand(len(self.history[0][1].cost))
            self.theta = theta / (np.sum(theta) + 1e-10)
        self._n_iter += 1

        pos = min(n // (self.n_trials // len(self.thresh_list)), len(self.thresh_list) - 1)
        self.thresh = self.thresh_list[pos]

        if self._rnd_branch_rng.random() < self.rnd_prob:
            return [self.configspace.sample_configuration() for _ in range(self.q)]

        configs = [ti.config for ti, _ in self.history]
        costs = np.array([np.atleast_1d(tv.cost) for _, tv in self.history], dtype=float)
        scalarized = self.scalarize(costs)
        self.incumbent = configs[int(np.argmin(scalarized))]

        selected: list[str] = []
        if self.thresh > 0 and self.hpi_method != 'no':
            selected = self.select(np.array([c.get_array() for c in configs]), scalarized)

        inc = self.config_to_vector(self.incumbent)
        fixed = {j: float(inc[j]) for j, name in enumerate(self.tunable) if name not in selected} if selected else {}
        X = np.array([self.config_to_vector(c) for c in configs])
        # never evaluate a configuration twice: integer snapping maps nearby candidates onto evaluated ones, which are
        # replaced by random points of the same subspace (the full space if it is exhausted)
        seen = set(configs)
        batch = []
        for x in self.candidate(X, costs, fixed):
            config = self.vector_to_config(x)
            tries = 0
            while config in seen:
                config = self.sample_subspace(fixed) if tries < 100 else self.configspace.sample_configuration()
                tries += 1
            seen.add(config)
            batch.append(config)
        return batch

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        self.history.append((trial_info, trial_value))

    def get_pareto_front(self) -> list[tuple[TrialInfo, TrialValue]]:
        costs = np.array([v[1].cost for v in self.history])
        return np.array(self.history, dtype=object)[pareto(costs)].tolist()

    def get_current_incumbent(self):
        if self.task.output_space.n_objectives == 1:
            return min(self.history, key=lambda x: x[1].cost)
        return self.get_pareto_front()
