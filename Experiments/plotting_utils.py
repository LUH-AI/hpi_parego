"""Plotting functions shared by the plotting notebooks of all experiments."""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from carps.analysis.gather_data import get_interpolated_performance_df, normalize_logs
from numpy import trapz
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon

from hpi_parego.igd_plus import add_igd_plus

COLORS = px.colors.qualitative.Plotly
METRICS = {
    "igd": ("igd_plus_inc_norm", "IGD+ regret"),
    "hv": ("trial_value__cost_inc_norm", "HV regret"),
    "soo": ("trial_value__cost_inc_norm", "incumbent regret"),
}
OPTIMIZERS = ["HPI-ParEGO", "Optuna-MO-NSGAII", "Optuna-MO-TPE", "Optuna-MO-EHVI", "Nevergrad-DE", "ParEGO"]
OPTIMIZER_COLORS = {opt: COLORS[i] for i, opt in enumerate(OPTIMIZERS)}
INTERPOLATION_COLUMNS = ["trial_value__cost_inc_norm", "igd_plus_inc_norm"]


def load_results(results_dirs, multi_objective=True, n_points=50, optimizers=None, tasks=None):
    """Reads the logs gathered with `python -m carps.analysis.gather_data <results_dir>` (one or several
    directories), optionally keeps only some optimizers and tasks (substrings of the task ids), normalizes them per task
    over all kept runs, adds IGD+ for multi-objective runs and interpolates all runs to n_points equally spaced trial
    fractions."""
    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]
    df = pd.concat([pd.read_parquet(os.path.join(d, "logs.parquet")) for d in results_dirs], ignore_index=True)
    if optimizers is not None:
        df = df[df["optimizer_id"].isin(optimizers)]
    if tasks is not None:
        df = df[df["task_id"].str.contains("|".join(tasks))]
    df = normalize_logs(df.reset_index(drop=True))
    if multi_objective:
        df = add_igd_plus(df)
    if "set" not in df.columns:
        df["set"] = "0"
    columns = INTERPOLATION_COLUMNS if multi_objective else INTERPOLATION_COLUMNS[:1]
    df = get_interpolated_performance_df(df, n_points=n_points, interpolation_columns=columns)
    return df[df["optimizer_id"].notna()]


def figure_dir(name):
    os.makedirs(name, exist_ok=True)
    return name


def hex_to_rgba(hex_color, alpha=0.1):
    """Converts a 6-digit hex color code (e.g., "#FF5733") to an RGBA color string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
    return f"rgba({r}, {g}, {b}, {alpha})"


def color_map(labels):
    """HPI variants (HPI-...) keep the color of HPI-ParEGO (blue), the other labels get the remaining colors in order."""
    if all(label in OPTIMIZER_COLORS for label in labels):
        return {label: OPTIMIZER_COLORS[label] for label in labels}
    rest = iter(COLORS[1:])
    return {label: COLORS[0] if str(label).startswith("HPI-") else next(rest) for label in sorted(labels, key=str)}


def mean_over_tasks(df, metric_col, group="optimizer_id"):
    """Mean of the metric over tasks per seed and trial fraction, then mean and standard error over seeds."""
    per_seed = df.groupby([group, "seed", "n_trials_norm"])[metric_col].mean()
    return (per_seed.reset_index().groupby([group, "n_trials_norm"])[metric_col]
            .agg(mean="mean", sem="sem").reset_index())


def add_curves(fig, curves, group, colors, row=None, col=None):
    order = curves[curves["n_trials_norm"] == curves["n_trials_norm"].max()].sort_values("mean")[group].tolist()
    for label in reversed(order):
        c = curves[curves[group] == label]
        kwargs = dict(row=row, col=col) if row else {}
        fig.add_trace(go.Scatter(x=c["n_trials_norm"], y=c["mean"], mode="lines", name=str(label),
                                 line=dict(width=2, color=colors[label])), **kwargs)
        fig.add_trace(go.Scatter(
            x=pd.concat([c["n_trials_norm"], c["n_trials_norm"][::-1]]),
            y=pd.concat([c["mean"] - c["sem"], (c["mean"] + c["sem"])[::-1]]),
            fill="toself", fillcolor=hex_to_rgba(colors[label], 0.25), line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", showlegend=False), **kwargs)


def plot_results(df, metric="igd", filename="results.png", group="optimizer_id", width=650, height=500):
    """Mean normalized regret (log-scale) over the normalized number of trials with the standard error over seeds,
    averaged over tasks."""
    metric_col, metric_name = METRICS[metric]
    curves = mean_over_tasks(df, metric_col, group)
    fig = go.Figure()
    add_curves(fig, curves, group, color_map(curves[group].unique()))
    fig.update_layout(height=height, width=width, template="plotly_white", font=dict(size=24),
                      legend=dict(font=dict(size=24), yanchor="top", y=1, xanchor="right", x=1,
                                  bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=0.5),
                      margin=dict(t=10, l=10, r=10, b=10))
    fig.update_xaxes(title_text="Normalized trials")
    fig.update_yaxes(type="log", title_text=f"Normalized mean<br>{metric_name} (log-scale)")
    fig.write_image(filename, width=width, height=height, scale=5)
    fig.show()


def plot_task_diff(df, metric="igd", filename="task_diff.png", reference="ParEGO", method="HPI-ParEGO"):
    """Per task: mean difference in normalized regret between reference and method with the standard error over
    seeds. Values > 0 indicate a better performance of the method."""
    metric_col, metric_name = METRICS[metric]
    keys = ["task_id", "seed", "n_trials_norm"]
    merged = pd.merge(df[df["optimizer_id"] == method][keys + [metric_col]],
                      df[df["optimizer_id"] == reference][keys + [metric_col]], on=keys, suffixes=("_m", "_r"))
    merged["diff"] = merged[f"{metric_col}_r"] - merged[f"{metric_col}_m"]
    merged["task"] = merged["task_id"].str.split("/").str[-2].where(merged["task_id"].str.contains("yahpo"),
                                                                    merged["task_id"].str.split("/").str[-1])
    curves = mean_over_tasks(merged, "diff", group="task")
    fig = go.Figure()
    add_curves(fig, curves, "task", {t: COLORS[i % len(COLORS)] for i, t in enumerate(sorted(curves["task"].unique()))})
    fig.add_hline(y=0, line=dict(color="gray", dash="dash"))
    fig.update_layout(height=500, width=1300, template="plotly_white", font=dict(size=24),
                      legend=dict(font=dict(size=24), yanchor="top", y=1.0, xanchor="left", x=1.01,
                                  bordercolor="gray", borderwidth=0.5),
                      margin=dict(t=10, l=10, r=10, b=10))
    fig.update_xaxes(title_text="Normalized trials")
    fig.update_yaxes(title_text=f"Δ in normalized mean<br>{metric_name} (vs. {reference})")
    fig.write_image(filename, width=1300, height=500, scale=5)
    fig.show()


def auc_per_run(df, metric_col, group="optimizer_id"):
    """AUC of the normalized regret over the normalized number of trials for every run (task x seed)."""
    return (df.groupby([group, "task_id", "seed"])
            .apply(lambda g: trapz(g.sort_values("n_trials_norm")[metric_col], g.sort_values("n_trials_norm")["n_trials_norm"]))
            .rename("auc").reset_index())


def plot_ablation(df, labels, metric="igd", filename="ablation.png", group="optimizer_id"):
    """Left: normalized regret over trials. Right: AUC per variant (mean over tasks, mean and standard error over
    seeds). labels maps the values of the group column to the names shown in the legend; the first one is
    HPI-ParEGO."""
    metric_col, metric_name = METRICS[metric]
    df = df[df[group].isin(labels)].copy()
    df[group] = df[group].map(labels)
    names = list(labels.values())
    colors = {name: COLORS[i % len(COLORS)] for i, name in enumerate(names)}
    curves = mean_over_tasks(df, metric_col, group)
    auc = auc_per_run(df, metric_col, group).groupby([group, "seed"])["auc"].mean().reset_index()
    auc = auc.groupby(group)["auc"].agg(mean="mean", sem="sem").reindex(names[::-1]).reset_index()

    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.05)
    add_curves(fig, curves, group, colors, row=1, col=1)
    fig.add_trace(go.Bar(x=auc["mean"], y=auc[group], orientation="h", error_x=dict(type="data", array=auc["sem"]),
                         marker=dict(color=[colors[n] for n in auc[group]], line=dict(color="black", width=1)),
                         showlegend=False), row=1, col=2)
    fig.update_layout(height=500, width=1300, template="plotly_white", font=dict(size=24),
                      legend=dict(font=dict(size=24), orientation="h", yanchor="bottom", y=1.02, xanchor="center",
                                  x=0.5), margin=dict(t=0, l=0, r=0, b=0))
    fig.update_xaxes(title_text="Normalized trials", row=1, col=1)
    fig.update_yaxes(type="log", title_text=f"Normalized mean<br>{metric_name} (log-scale)", row=1, col=1)
    fig.update_xaxes(title_text=f"AUC of normalized<br>mean {metric_name}", row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.write_image(filename, width=1300, height=500, scale=5)
    fig.show()


def significance_vs_reference(df, metric_col, reference, group="optimizer_id"):
    """Paired Wilcoxon signed-rank test of each variant against the reference on the AUC of every run (task x seed),
    Holm-corrected over all variants. Returns the mean and standard error of the paired AUC difference
    (negative = better than the reference), the number of runs in which the variant is better (wins) and the adjusted
    p-value."""
    auc = auc_per_run(df, metric_col, group).pivot_table(index=["task_id", "seed"], columns=group, values="auc")
    raw = {}
    for variant in auc.columns.drop(reference):
        diff = (auc[variant] - auc[reference]).dropna()
        raw[variant] = dict(mean_diff=diff.mean(), sem=diff.sem(), wins=f"{int((diff < 0).sum())}/{len(diff)}",
                            p=wilcoxon(diff).pvalue if (diff != 0).any() else 1.0)
    # Holm: the i-th smallest p-value is multiplied by (m - i), adjusted values are kept monotone
    running = 0.0
    for i, variant in enumerate(sorted(raw, key=lambda v: raw[v]["p"])):
        running = max(running, min(1.0, (len(raw) - i) * raw[variant]["p"]))
        raw[variant]["p_holm"] = running
    return pd.DataFrame.from_dict(raw, orient="index")


def plot_significance(df, labels, metric="igd", filename="ablation_all.png", reference="HPI-ParEGO",
                      group="optimizer_id"):
    """Mean difference in AUC to the reference per variant (paired by task and seed) with the standard error.
    * marks a significant difference (paired Wilcoxon signed-rank test, Holm-corrected over all variants, p < 0.05)."""
    metric_col, metric_name = METRICS[metric]
    df = df[df[group].isin(labels)].copy()
    df[group] = df[group].map(labels)
    sig = significance_vs_reference(df, metric_col, reference, group).sort_values("mean_diff", ascending=False)
    print(sig.to_string(float_format=lambda v: f"{v:.4g}"))
    fig = go.Figure(go.Scatter(
        x=sig["mean_diff"], y=sig.index, mode="markers", showlegend=False,
        marker=dict(size=11, color=[COLORS[(i + 1) % len(COLORS)] for i in range(len(sig))],
                    line=dict(color="black", width=1)),
        error_x=dict(type="data", array=sig["sem"], color="black", thickness=1.5, width=6)))
    for variant, row in sig.iterrows():
        fig.add_annotation(x=row["mean_diff"] + row["sem"], y=variant, text="*" if row["p_holm"] < 0.05 else "n.s.",
                           showarrow=False, xanchor="left", xshift=6, font=dict(size=16))
    fig.add_vline(x=0, line=dict(color="gray", dash="dash"))
    fig.update_layout(height=600, width=550, template="plotly_white", font=dict(size=18),
                      margin=dict(t=20, l=0, r=50, b=0),
                      xaxis_title=f"Δ AUC of normalized {metric_name}<br>vs. {reference} (← better)")
    fig.write_image(filename, width=550, height=400, scale=5)
    fig.show()


def plot_scalability(df, metric="igd", filename="scalability.png"):
    """AUC of the normalized regret per optimizer and number of hyperparameters (mean and standard error over seeds)."""
    metric_col, metric_name = METRICS[metric]
    auc = auc_per_run(df, metric_col).merge(df[["task_id", "dim"]].drop_duplicates(), on="task_id")
    auc = auc.groupby(["optimizer_id", "dim"])["auc"].agg(mean="mean", sem="sem").reset_index().sort_values("dim")
    colors = color_map(auc["optimizer_id"].unique())
    fig = go.Figure([go.Bar(name=opt, x=g["dim"].astype(str), y=g["mean"], error_y=dict(type="data", array=g["sem"]),
                            marker=dict(color=colors[opt], line=dict(color="black", width=1)))
                     for opt, g in auc.groupby("optimizer_id")])
    fig.update_layout(barmode="group", template="plotly_white", font=dict(size=24), margin=dict(t=0, l=0, r=0, b=0),
                      xaxis_title="Number of hyperparameters", yaxis_title=f"AUC of normalized<br>mean {metric_name}",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    fig.write_image(filename, width=1300, height=500, scale=5)
    fig.show()
