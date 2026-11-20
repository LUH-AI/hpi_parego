from __future__ import annotations
import random
from types import SimpleNamespace
import numpy as np
from ConfigSpace import Constant, Configuration, ConfigurationSpace
from ConfigSpace.util import deactivate_inactive_hyperparameters
from hypershap import ExplanationTask, HyperSHAP


class ModelWrapper:
    """
    Wraps a model for HyperSHAP, which maximizes, so predict() returns the negated predicted mean.
    It takes arrays of the reduced configuration space (without constants) and fills in the remaining columns of the
    full configuration space, as the model expects the full width.
    """
    def __init__(self, model, configspace, reduced_cs):
        self.model = model
        names = list(configspace.keys())
        self.columns = [names.index(n) for n in reduced_cs.keys()]
        self.template = configspace.get_default_configuration().get_array()

    def predict(self, X: np.ndarray) -> np.ndarray:
        full = np.tile(self.template, (len(X), 1))
        full[:, self.columns] = np.asarray(X)
        mean = np.asarray(self.model.predict(full)[0])
        return (-1) * (mean[:, 0] if mean.ndim == 2 else mean)


class GPImportanceModel:
    """
    Paper Appendix Section D.1
    Gaussian Process (as in SMAC's BlackBox facade) fitted to the observed costs, scalarized with the current ParEGO
    weights, which HyperSHAP explains. The random forest used for optimization reproduces its best training point
    almost exactly and rarely predicts an improvement near it, whereas a GP captures small differences around the
    incumbent. The GP of the last step is refitted, so its kernel hyperparameter optimization starts from its last fit.
    """
    def __init__(self, configspace, X, y, gp=None, seed=0):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        self.gp = gp if gp is not None else make_smac_gp(configspace, seed)
        self.gp.train(X[ok], y[ok].reshape(-1, 1))

    def predict(self, X):
        return self.gp.predict(np.asarray(X, dtype=float))

    predict_marginalized = predict


def make_smac_gp(configspace, seed=0):
    """SMAC's GP as in its BlackBox facade (Matern kernel, Hamming kernel for categorical hyperparameters)."""
    from smac.facade.blackbox_facade import BlackBoxFacade
    return BlackBoxFacade.get_model(SimpleNamespace(configspace=configspace, seed=seed))


class LazyObjective:
    """
    Paper Appendix Section J
    Holds the task's objective function config and builds a separate instance (without loggers, so its evaluations
    are not logged as trials) only when the ground-truth importance is used.
    """
    def __init__(self, cfg):
        from omegaconf import OmegaConf
        self.cfg = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
        self._objective = None

    def get(self):
        if self._objective is None:
            from hydra.utils import instantiate
            self._objective = instantiate(self.cfg)
        return self._objective


class OracleModel:
    """
    Paper Appendix Section J
    Replaces the model explained by HyperSHAP by the true objective function of the benchmark (ground-truth
    importance). scalarize maps an (n, n_objectives) cost array to the cost the importance model is trained on.
    """
    def __init__(self, objective, configspace, scalarize):
        self.objective = objective
        self.cs = configspace
        self.scalarize = scalarize
        self.hps = list(configspace.values())
        names = list(configspace.keys())
        conditions = sorted(configspace.conditions, key=lambda c: names.index(c.child.name))
        self.conditions = [(names.index(c.child.name),
                            [names.index(getattr(p, "name", p)) for p in configspace.parents_of[c.child.name]], c)
                           for c in conditions]

    def predict(self, X):
        y = self.scalarize(self.costs(np.asarray(X, dtype=float)))
        return y.reshape(-1, 1), np.zeros((len(y), 1))

    predict_marginalized = predict

    def values(self, X):
        """Values per hyperparameter (None where inactive). Inactive children with a value are dropped and active
        ones without a value (fixed while their parent varies) get their default."""
        X = X.copy()
        active = np.ones(X.shape, dtype=bool)
        for child, parents, cond in self.conditions:
            for p in parents:
                X[~active[:, p], p] = np.nan
            filled = np.where(np.isnan(X), 0.0, X)
            active[:, child] = active[:, parents].all(1) & cond.satisfied_by_vector_array(filled.T)
            missing = active[:, child] & np.isnan(X[:, child])
            X[missing, child] = self.hps[child].to_vector(self.hps[child].default_value)
        X[~active] = np.nan
        out = []
        for i, hp in enumerate(self.hps):
            ok = ~np.isnan(X[:, i])
            col = np.full(len(X), None, dtype=object)
            if ok.any():
                col[ok] = list(hp.to_value(X[ok, i]))
            out.append(col)
        return out

    def rows(self, X):
        names = [hp.name for hp in self.hps]
        return [{n: (v.item() if hasattr(v, 'item') else v) for n, v in zip(names, r) if v is not None}
                for r in zip(*self.values(X))]

    def costs(self, X):
        obj = self.objective
        if hasattr(obj, '_problem'):  # PyMOO, vectorized
            cols = dict(zip((hp.name for hp in self.hps), self.values(X)))
            return np.asarray(obj._problem.evaluate(np.column_stack(
                [cols[f"x{i}"].astype(float) for i in range(obj._problem.n_var)])), dtype=float).reshape(len(X), -1)
        rows = self.rows(X)
        if hasattr(obj, '_objective_function') and hasattr(obj, 'metrics'):  # YAHPO-Gym, as in its _evaluate
            from carps.objective_functions.yahpo import maybe_invert
            fidelities = dict(getattr(obj, 'max_other_fidelities', {}))
            if obj.budget_type is not None:
                fidelities[obj.budget_type] = obj.fidelity_space[obj.budget_type].upper
            out = obj._objective_function.objective_function([{**r, **fidelities} for r in rows])
            return np.array([[maybe_invert(o[m], m) for m in obj.metrics] for o in out], dtype=float)
        from carps.utils.trials import TrialInfo
        return np.array([np.atleast_1d(obj.evaluate(TrialInfo(config=Configuration(self.cs, values=r))).cost)
                         for r in rows], dtype=float)


class PseudoConstant(Constant):
    """
    Creates a Constant as Hyperparameter without checking if the value is legal to avoid errors with the runhistory when the configuration space is adjusted.
    """
    def legal_value(self, value):
        return True


def tunable_names(configspace):
    """Hyperparameters that can be tuned: no constants (e.g. task ids) and no categoricals with a single choice."""
    return [hp.name for hp in configspace.values()
            if not isinstance(hp, Constant) and len(getattr(hp, 'choices', (0, 1))) > 1]


def remove_constants(configspace):
    """Copy of the configuration space without constants, which would only double the number of coalitions."""
    reduced = ConfigurationSpace()
    reduced.random.set_state(configspace.random.get_state())
    reduced.add([hp for hp in configspace.values() if not isinstance(hp, Constant)])
    reduced.add(configspace.conditions)
    return reduced


def fill_up(selected, configspace, size, rng=random):
    """Adds randomly chosen tunable hyperparameters until the selection has the given size."""
    rest = [n for n in tunable_names(configspace) if n not in selected]
    return list(selected) + rng.sample(rest, min(max(size - len(selected), 0), len(rest)))


def sum_hps_concise(hps_dict, threshold):
    """
    Paper Section 3.2
    Sums the Shapley values until the sum surpasses the threshold times the cumsum. Hyperparameters contributing less
    than min(5%, 1 - threshold) of the total are not selected. An interaction term brings in all of its
    hyperparameters.
    Args:
        hps_dict (dict): A dictionary mapping tuples of hyperparameter names to their Shapley values.
        threshold (float): The threshold for selecting important hyperparameters.
    Returns:
        list: A list of selected important hyperparameter names.
    """
    sorted_items = sorted([(v, k) for k, v in hps_dict.items() if v > 0 and len(k) > 0], reverse=True)

    hps = np.array([v for v, k in sorted_items])
    total_sum = hps.sum()
    min_imp = min(0.05, (1-threshold))
    cumulative_target = threshold * total_sum
    min_val = min_imp * total_sum

    selected_hps = []
    cum_sum = 0.0

    for hp_value, key in sorted_items:
        if cum_sum < cumulative_target and hp_value > min_val:
            selected_hps += [n for n in key if n not in selected_hps]
            cum_sum += hp_value
        elif cum_sum >= cumulative_target:
            break
    return selected_hps


def select_at_least_two(scores, configspace, thresh, rng=random):
    """
    Paper Section 3.3
    Selection by the threshold rule. If no hyperparameter is important, nothing is selected and the original space is
    used; a single selected hyperparameter is complemented by a random one.
    """
    # Shapley values of zero come out as tiny positive or negative numbers due to rounding, which must not count as
    # an improvement
    tol = max(1e-12, 1e-9 * max((abs(v) for v in scores.values()), default=0.0))
    selected = sum_hps_concise({k: (v if abs(v) > tol else 0.0) for k, v in scores.items()}, threshold=thresh)
    if len(selected) == 0:
        return selected
    return fill_up(selected, configspace, max(2, len(selected)), rng)


def calculate_hpi_hypershap(model, configspace, thresh, ref_cfg, rng=random, interactions=False):
    """
    Paper Section 3.2 and Appendix Section D.1
    Calculates the HPI based on the HyperSHAP tunability game and returns the important hyperparameters.
    For each coalition, HyperSHAP samples 2,000 configurations in which only the coalition varies from the reference
    configuration. In fast mode, only coalitions of at most three hyperparameters are simulated. Above 14
    hyperparameters, the Shapley values are approximated with 10,000 coalitions.

    Args:
        model: model to explain (GP, surrogate model, or ground truth)
        configspace (ConfigSpace): The configuration space containing hyperparameters.
        thresh (float): current threshold
        ref_cfg: reference configuration for hypershap
        rng: random number generator for completing a single selected hyperparameter
        interactions (bool): use the order-2 k-SII (main effects and pairs) instead of the Shapley values

    Returns:
        list: list of important hps
        dict: Shapley values
    """
    reduced = remove_constants(configspace)
    # The model is explained directly, as `from_function` would fit a proxy forest on uniform samples, which loses
    # the structure around the incumbent.
    et = ExplanationTask.from_base_model(reduced, ModelWrapper(model, configspace, reduced))

    def to_reduced(cfg):
        # configs in the runhistory can carry values for inactive conditional hyperparameters, which a validated
        # Configuration rejects; they have no effect, so drop them
        return deactivate_inactive_hyperparameters({k: v for k, v in dict(cfg).items() if k in reduced}, reduced)

    order, index = (2, 'k-SII') if interactions else (1, 'SV')
    try:
        hypershap = HyperSHAP(et, approximation_budget=10_000)
        hypershap.tunability(baseline_config=to_reduced(ref_cfg), order=order, index=index, n_samples=2000,
                             fast_mode=True)
    except Exception:
        print('Using default configuration as baseline for hypershap!')
        hypershap = HyperSHAP(et, approximation_budget=10_000)
        hypershap.tunability(baseline_config=to_reduced(configspace.get_default_configuration()), order=order,
                             index=index, n_samples=2000, fast_mode=True)
    sv = dict(hypershap.get_interaction_values_with_names())
    return select_at_least_two(sv, configspace, thresh, rng), sv


def calculate_hpi_history(free_hps, costs, configspace, thresh, rng=random):
    """
    Paper Appendix Section D.2
    Selects the important hyperparameters with the criterion of MCTS-VS (Song et al., 2022): a hyperparameter scores
    the mean of the (negated) scalarized observed costs of the configurations in which it was tuned, relative to the
    mean over all configurations. The costs are scalarized with the current weights, so the scores follow the
    scalarization. The selection then works as for HyperSHAP.

    Args:
        free_hps: one set of tuned hyperparameter names per configuration
        costs: scalarized observed costs per configuration (inf for crashed ones)
        configspace (ConfigSpace): The configuration space containing hyperparameters.
        thresh (float): current threshold
        rng: random number generator for completing a single selected hyperparameter
    Returns:
        list: list of important hps
        dict: scores
    """
    costs = np.asarray(costs, dtype=float)
    ok = np.isfinite(costs)
    score = -costs[ok]
    free = [f for f, o in zip(free_hps, ok) if o]
    scores = {}
    for name in tunable_names(configspace):
        tuned = np.array([name in f for f in free], dtype=bool)
        scores[(name,)] = float(score[tuned].mean() - score.mean()) if tuned.any() else 0.0
    return select_at_least_two(scores, configspace, thresh, rng), scores


def random_selection(configspace, thresh, rng=random):
    """
    Paper Appendix Section D.2
    Randomly selects ceil((1 - thresh) * n) of the n tunable hyperparameters (constants such as task ids excluded).

    Args:
        configspace (ConfigSpace): The configuration space containing hyperparameters.
        thresh (float): The threshold for selecting hyperparameters.
        rng: random number generator
    Returns:
        list: A list of selected hyperparameter names.
    """
    hps = tunable_names(configspace)
    n_hps = int(np.ceil((1 - thresh) * len(hps)))
    return rng.sample(hps, n_hps)
