"""
ALS ablation/evaluation on frozen cross-validation folds.

This module evaluates a family of ALS matrix factorization models derived from
a tuned baseline (“best params”) and reports how different components
contribute to performance. It runs each ablation across the same precomputed
folds, computes overall RMSE, RMSE per item-popularity bin, timing,
early-stopping diagnostics, and significance vs the baseline. It also emits a
set of comparative plots and convergence curves.

You must provide:
- a ratings matrix `R.npy`
- frozen folds produced by `scripts/create_folds.py` (an `.npz`)
- a JSON with best params (if not available, run `scripts/tune_params.py` first)
- optional item features used by ALS (e.g., genres, years), as a dict of
  name -> ndarray

## What gets evaluated

Given `best_params`, we generate a grid of named variants:
- `full`              — baseline (best params as-is)
- `no_features`       — all λ_w_* set to 0
- `only_<feature>`    — keep λ_w_<feature> from baseline;
                        others → 0 (for features used in baseline)
- `no_graph`          — disable graph regularization
- `graph_feature=<f>` — swap graph source feature to `<f>` while keeping `alpha`
- `no_pop_reg`        — disable popularity regularization

(Variants that don't apply to the baseline are skipped automatically, and
duplicate parameterizations are deduplicated.)

## Artifacts (under: `<out_dir>/ablations`):

- `ablations.csv`              — table of metrics per variant:
    - rmse_mean, rmse_std, time_mean, time_std
    - rmse_pop_1..rmse_pop_K (per popularity bin)
    - p_raw, p_fdr (sign test vs `full`, with BH-FDR)
    - delta_mean (mean RMSE difference vs `full`, negative is better)
    - early-stopping diagnostics + key params for traceability
- `ablations.json`             — metadata + the same results as JSON
- `plots/`
    - `rmse_bar_vertical.png`  — RMSE by variant (mean ± std)
    - `time_bar_vertical.png`  — training time by variant (mean ± std)
    - `rmse_vs_time.png`       — scatter with error bars
    - `bins_grouped_bars.png`  — RMSE per popularity bin (grouped bars)
    - `bins_heatmap.png`       — ΔRMSE vs baseline per popularity bin
- `convergence/`
    - `<variant>.json`         — per-variant mean/std train RMSE by iteration
    - `convergence_all.png`    — all variants' convergence curves

## Typical usage

```python
from scripts.ablation import run_ablation
import numpy as np

R_PATH = "data/ratings.npy"
FOLDS_PATH = "artifacts/folds/entrywise_5_fold_seed_42.npz"
BEST_PARAMS_PATH = "results/tuning/als_tune_v1_best_params.json"

# Example item features
genres = np.load("data/genres.npy")
years  = np.load("data/years.npy").reshape(-1, 1)
features = {"genres": genres, "years": years}

csv_path, json_path = run_ablation(
    R_path=R_PATH,
    folds_path=FOLDS_PATH,
    best_params_path=BEST_PARAMS_PATH,
    features=features,
    out_dir="results",
    n_pop_bins=5,
    es_tol=None,        # defaults to tuning ES settings
    es_min_iters=None,  # defaults to tuning ES settings
    verbose_fit=0,
)

print("Ablations CSV:", csv_path)
print("Ablations JSON:", json_path)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.als import ALS
from scripts.create_folds import load_folds_npz, make_train_valid_split
from scripts.tune_params import (
    _make_config,
    _normalize_params,
    _rmse_on_indices,
    ES_TOL as TUNE_ES_TOL,
    ES_MIN_ITERS as TUNE_ES_MIN_ITERS,
    DEFAULT_RANDOM_STATE
)

# Popularity binning
N_POP_BINS: int = 5
POP_BIN_STRATEGY: str = "quantile"  # ["quantile", "uniform"]


@dataclass
class AblationResultRow:
    """One ablation variant's aggregated results across CV folds."""
    variant: str
    rmse_mean: float
    rmse_std: float
    time_mean: float
    time_std: float
    mean_iters: float
    early_stopped_folds: int
    target_n_iters: int
    es_tol: float
    es_min_iters: int
    rmse_bins: Dict[str, float]
    params: Dict[str, Any]
    p_raw: Optional[float] = None
    p_fdr: Optional[float] = None
    delta_mean: Optional[float] = None


def _popularity_bins_from_R(
    R: np.ndarray,
    n_bins: int = N_POP_BINS,
    strategy: str = POP_BIN_STRATEGY
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute item popularity bins from the ratings matrix.

    Args:
        R: (m x n) rating matrix with NaN for missing entries.
        n_bins: Number of popularity bins.
        strategy: Binning strategy, either 'quantile' or 'uniform'.

    Returns:
        item_bin: (n,) array of bin indices (0 to n_bins-1) per item.
        edges: (n_bins + 1,) array of bin edges.
    """
    counts = np.sum(~np.isnan(R), axis=0).astype(float)

    if strategy == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(counts, qs)
    elif strategy == "uniform":
        edges = np.linspace(
            float(np.min(counts)), float(np.max(counts)), n_bins + 1
            )
    else:
        raise ValueError(f"Unknown popularity binning strategy '{strategy}'")

    # Ensure strictly increasing edges to avoid degenerate bins
    edges = np.array(edges, dtype=float)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    # Assign bins (rightmost inclusive)
    item_bin = np.clip(np.searchsorted(
        edges, counts, side="right") - 1, 0, n_bins - 1
                       )
    return item_bin.astype(int), edges


def _split_val_indices_by_popularity(
    val_idx: np.ndarray,
    shape: Tuple[int, int],
    item_bin: np.ndarray,
    n_bins: int
) -> List[np.ndarray]:
    """Split validation flat indices by item popularity bin.

    Args:
        val_idx: (k,) array of flat validation indices into R.
        shape: Shape of the ratings matrix R (m, n).
        item_bin: (n,) array of bin indices (0 to n_bins-1) per item.
        n_bins: Number of popularity bins.

    Returns:
        List of n_bins arrays of flat indices for each popularity bin.
    """
    _, n = shape
    cols = val_idx % n
    return [val_idx[item_bin[cols] == b] for b in range(n_bins)]


def _eval_variant_cv(
    variant_name: str,
    R: np.ndarray,
    features: Dict[str, np.ndarray],
    folds: List[np.ndarray],
    params: Dict[str, Any],
    item_bin: np.ndarray,
    n_pop_bins: int,
    es_tol: float,
    es_min_iters: int,
    convergence_curves: Dict[str, List[List[float]]],
    verbose_fit: int = 0,
) -> Tuple[List[float], List[float], List[Dict[str, float]], List[int]]:
    """Evaluate a fixed-parameter model across CV folds and record convergence.

    Args:
        variant_name: Name of the ablation variant.
        R: (m x n) rating matrix with NaN for missing entries.
        features: Dict of item side features.
        folds: List of K arrays of flat indices into R used as validation.
        params: Model hyperparameters for this variant.
        item_bin: (n,) array of bin indices (0 to n_pop_bins-1) per item.
        n_pop_bins: Number of popularity bins.
        es_tol: Early-stopping tolerance.
        es_min_iters: Minimum iterations before early stopping.
        convergence_curves: Dict to store per-variant convergence curves.
        verbose_fit: Verbosity level for ALS fitting.

    Returns:
        fold_rmse: List of RMSE values per fold.
        fold_time: List of training times per fold.
        fold_bin_rmse: List of dicts of bin-wise RMSE per fold.
        fold_iters: List of number of iterations per fold.
    """
    # Normalize params to data
    params = _normalize_params(dict(params), R.shape, list(features.keys()))
    cfg = _make_config(params)

    # Map λ_w per feature for ALS
    lambda_w_map = {
        name: float(
            params.get(f"lambda_w_{name}", 0.0)
            ) 
        for name in features.keys()
        }

    fold_rmse, fold_time, fold_bin_rmse, fold_iters = [], [], [], []

    for i, _ in enumerate(folds):
        R_train, R_valid, val_idx = make_train_valid_split(R, folds, i)

        t0 = time.perf_counter()
        model = ALS(config=cfg, lambda_w=lambda_w_map)
        model.fit(
            R_train,
            features=features,
            tol=es_tol,
            min_iters=es_min_iters,
            verbose=verbose_fit
            )
        R_hat = model.predict(features=features)
        t1 = time.perf_counter()

        # Keep convergence curve
        convergence_curves.setdefault(variant_name, []).append(
            list(model.history.get("train_rmse", []))
        )

        # Overall fold metrics
        fold_rmse.append(_rmse_on_indices(R_valid, R_hat, val_idx))
        fold_time.append(t1 - t0)
        fold_iters.append(len(model.history.get("train_rmse", [])))

        # Popularity-bin RMSEs
        bin_indices = _split_val_indices_by_popularity(
            val_idx, R.shape, item_bin, n_pop_bins
            )
        fold_bin_rmse.append({
            f"rmse_pop_{b+1}": _rmse_on_indices(R_valid, R_hat, idx_b)
            for b, idx_b in enumerate(bin_indices)
        })

    return fold_rmse, fold_time, fold_bin_rmse, fold_iters


def _aggregate_convergence(curves: List[List[float]]) -> Dict[str, Any]:
    """Aggregate per-fold convergence curves into mean/std by iteration.

    Args:
        curves: List of per-fold RMSE curves (list of floats).

    Returns:
        Dict with keys:
            - "iters": List of iteration numbers (1-based).
            - "rmse_mean": List of mean RMSE per iteration.
            - "rmse_std": List of stddev RMSE per iteration.
            - "n_folds": Number of folds aggregated.
    """
    # Pad curves to equal length with NaNs
    if not curves:
        return {"iters": [], "rmse_mean": [], "rmse_std": [], "n_folds": 0}
    maxlen = max(len(c) for c in curves)
    arr = np.full((len(curves), maxlen), np.nan, dtype=float)
    for j, c in enumerate(curves):
        arr[j, :len(c)] = np.asarray(c, dtype=float)

    return {
        "iters": list(range(1, maxlen + 1)),
        "rmse_mean": np.nanmean(arr, axis=0).tolist(),
        "rmse_std": np.nanstd(arr, axis=0).tolist(),
        "n_folds": len(curves),
    }


def _aggregate_bins_mean(
    fold_bin_rmse: List[Dict[str, float]]
) -> Dict[str, float]:
    """Average bin-wise RMSE across folds.

    Args:
        fold_bin_rmse: List of dicts of bin-wise RMSE per fold.

    Returns:
        Dict of mean bin-wise RMSE across folds.
    """
    # Handle empty case
    if not fold_bin_rmse:
        return {}

    # Get all bin keys
    keys = sorted(fold_bin_rmse[0].keys())
    
    return {k: float(np.nanmean([d[k] for d in fold_bin_rmse])) for k in keys}


def _sign_test_paired(x: List[float], y: List[float]) -> float:
    """Two-sided paired sign test (exact binomial), robust for small folds.

    Args:
        x: First list of values.
        y: Second list of values.

    Returns:
        Two-sided p-value.
    """
    diffs = [a - b for a, b in zip(x, y) if not np.isclose(a - b, 0.0)]
    n = len(diffs)
    if n == 0:
        return 1.0
    k_pos = sum(d > 0 for d in diffs)

    def binom_cdf(k: int, n_: int) -> float:
        s = 0.0
        for i in range(0, k + 1):
            s += math.comb(n_, i)
        return s / (2 ** n_)

    cdf = binom_cdf(k_pos, n)
    sf = 1.0 - binom_cdf(k_pos - 1, n) if k_pos > 0 else 1.0

    return float(min(1.0, 2.0 * min(cdf, sf)))


def _fdr_bh(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction.

    Args:
        pvals: List of p-values.

    Returns:
        List of adjusted p-values.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    adj = np.array(
        [p * m / i for i, p in enumerate(ranked, start=1)], dtype=float
        )
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    out = np.empty(m, dtype=float)
    out[order] = np.clip(adj, 0.0, 1.0)

    return out.tolist()


def _variant_grid(
    best_params: Dict[str, Any],
    feature_names: List[str]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Enumerate ablation variants (baseline + controlled removals/additions).

    Args:
        best_params: Baseline best parameters.
        feature_names: List of item feature names.

    Returns:
        List of (variant_name, params) tuples.
    """
    variants: List[Tuple[str, Dict[str, Any]]] = []

    # Baseline (full)
    base = dict(best_params)
    variants.append(("full", base))

    # Determine toggles present in baseline
    alpha = float(base.get("alpha", 0.0))
    graph_enabled = (
        alpha > 0.0 and base.get("graph_feature", "__none__") in feature_names
        )
    pop_on = base.get("pop_reg_mode", None) is not None
    feat_used = {
        f: float(base.get(f"lambda_w_{f}", 0.0)) > 0.0 for f in feature_names
        }
    any_feat_on = any(feat_used.values())

    # Remove all features
    if any_feat_on:
        p = dict(base)
        for f in feature_names:
            p[f"lambda_w_{f}"] = 0.0
        variants.append(("no_features", p))

        # Keep only single feature (for those in use)
        for f in feature_names:
            if feat_used[f]:
                p2 = dict(base)
                for g in feature_names:
                    p2[f"lambda_w_{g}"] = 0.0
                p2[f"lambda_w_{f}"] = float(base.get(f"lambda_w_{f}", 0.0))
                variants.append((f"only_{f}", p2))

    # Remove graph
    if graph_enabled:
        p = dict(base)
        p["alpha"] = 0.0
        p["graph_feature"] = "__none__"
        variants.append(("no_graph", p))

        # Try alternative graph source features
        for f in feature_names:
            if f != base.get("graph_feature"):
                p2 = dict(base)
                p2["alpha"] = alpha
                p2["graph_feature"] = f
                variants.append((f"graph_feature={f}", p2))

    # Remove popularity regularization
    if pop_on:
        p = dict(base)
        p["pop_reg_mode"] = None
        variants.append(("no_pop_reg", p))

    # Deduplicate by parameter signature
    uniq: Dict[Tuple, Tuple[str, Dict[str, Any]]] = {}
    for name, p in variants:
        key = tuple(sorted(p.items()))
        uniq[key] = (name, p)

    return list(uniq.values())


def _safe_savefig(fig, path: Path) -> None:
    """Safely save a Matplotlib figure to PNG and close it.

    Args:
        fig: Matplotlib figure.
        path: Path to save the figure.
    """
    try:
        fig.savefig(path.as_posix(), dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)


def _save_comparative_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Create comparative PNG plots from the ablation summary DataFrame.
    Produces:
      - rmse_bar.png
      - time_bar.png
      - rmse_vs_time.png
      - bins_heatmap.png
      - bins_grouped_bars.png
    """
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _baseline_index(frame: pd.DataFrame) -> Optional[int]:
        if "full" in frame["variant"].values:
            return int(np.where(frame["variant"].values == "full")[0][0])
        return None

    def _grid(ax):
        ax.grid(True, linewidth=0.6, alpha=0.35, zorder=0)

    # RMSE barplot
    df_bar = df.sort_values("rmse_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(df_bar)), 5))
    x = np.arange(len(df_bar))
    ax.bar(x, df_bar["rmse_mean"], yerr=df_bar["rmse_std"], alpha=0.95, zorder=2)
    ax.set_xticks(x, df_bar["variant"], rotation=45, ha="right")
    ax.set_ylabel("RMSE (mean ± std)")
    ax.set_title("Ablation — RMSE by variant")
    _grid(ax)
    bidx = _baseline_index(df_bar)
    if bidx is not None:
        ax.bar(bidx, df_bar.loc[bidx, "rmse_mean"], yerr=df_bar.loc[bidx, "rmse_std"],
               color="tab:orange", alpha=0.95, zorder=3)
    _safe_savefig(fig, plots_dir / "rmse_bar.png")

    # Time barplot
    df_time = df_bar
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(df_time)), 5))
    ax.bar(
        x, df_time["time_mean"], yerr=df_time["time_std"], alpha=0.95, zorder=2
        )
    ax.set_xticks(x, df_time["variant"], rotation=45, ha="right")
    ax.set_ylabel("Time (s) mean ± std")
    ax.set_title("Ablation — Training time by variant")
    _grid(ax)
    if bidx is not None:
        ax.bar(
            bidx, df_time.loc[bidx, "time_mean"],
            yerr=df_time.loc[bidx, "time_std"],
            color="tab:orange",
            alpha=0.95,
            zorder=3
            )
    _safe_savefig(fig, plots_dir / "time_bar.png")

    # RMSE vs Time scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(df["time_mean"], df["rmse_mean"],
                xerr=df["time_std"], yerr=df["rmse_std"],
                fmt="o", alpha=0.95, zorder=2)
    ax.set_xlabel("Time (s) mean ± std")
    ax.set_ylabel("RMSE mean ± std")
    ax.set_title("Ablation — RMSE vs Time")
    _grid(ax)
    for _, r in df.sort_values("rmse_mean", ascending=False).iterrows():
        ax.annotate(
            r["variant"],
            (r["time_mean"], r["rmse_mean"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8
            )
    _safe_savefig(fig, plots_dir / "rmse_vs_time.png")

    # Popularity-bin comparisons
    bin_cols = [c for c in df.columns if c.startswith("rmse_pop_")]
    if bin_cols:
        df_top = df.sort_values("rmse_mean").reset_index(drop=True)
        V = len(df_top)
        B = len(bin_cols)
        width = max(0.75 / B, 0.08)

        fig, ax = plt.subplots(figsize=(max(8, 0.7 * B * V / 3), 5))
        base_x = np.arange(B)
        for i, (_, row) in enumerate(df_top.iterrows()):
            offsets = base_x + (i - (V - 1) / 2) * width
            heights = [row[c] for c in bin_cols]
            ax.bar(
                offsets,
                heights,
                width=width,
                label=row["variant"],
                alpha=0.9,
                zorder=2
                )

        ax.set_xticks(
            base_x,
            [c.replace("rmse_pop_", "Bin ") for c in bin_cols]
            )
        ax.set_ylabel("RMSE (per popularity bin)")
        ax.set_title("Ablation — RMSE per popularity bin")
        _grid(ax)
        _safe_savefig(fig, plots_dir / "bins_grouped_bars.png")

    # Heatmap: per-bin ΔRMSE vs baseline
    if "full" in df["variant"].values and bin_cols:
        baseline_bins = df.loc[df["variant"] == "full", bin_cols].iloc[0]
        mat, labels = [], []
        for _, r in df.iterrows():
            deltas = [
                (r[c] - baseline_bins[c]) if np.isfinite(r[c]) else np.nan
                for c in bin_cols
            ]
            mat.append(deltas); labels.append(r["variant"])
        mat = np.array(mat, dtype=float)

        fig, ax = plt.subplots(
            figsize=(1.2 * len(bin_cols) + 2, 0.45 * len(labels) + 2)
            )
        im = ax.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            zorder=2
            )
        ax.set_xticks(
            np.arange(len(bin_cols)),
            [c.replace("rmse_pop_", "Bin ") for c in bin_cols],
            rotation=45,
            ha="right"
            )
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_title("Δ RMSE vs baseline (per popularity bin)")
        vmax = np.nanmax(np.abs(mat))
        if np.isfinite(vmax) and vmax > 0:
            im.set_clim(-vmax, vmax)
        fig.colorbar(im, ax=ax, shrink=0.8)
        _safe_savefig(fig, plots_dir / "bins_heatmap.png")


def _save_convergence_artifacts(
    curves_per_variant: Dict[str, List[List[float]]],
    out_dir: Path,
    baseline_name: str = "full",
) -> None:
    """Save per-variant convergence JSON and a single PNG with averaged curves.

    Args:
        curves_per_variant: Dict of per-variant lists of per-fold RMSE curves.
        out_dir: Output directory to save artifacts.
        baseline_name: Name of the baseline variant to highlight.
    """
    conv_dir = out_dir / "convergence"
    conv_dir.mkdir(parents=True, exist_ok=True)

    # Per-variant JSON
    agg: Dict[str, Dict[str, Any]] = {}
    for variant, curves in curves_per_variant.items():
        agg[variant] = _aggregate_convergence(curves)
        with open(conv_dir / f"{variant}.json", "w") as f:
            json.dump(agg[variant], f, indent=2)

    # Combined figure
    fig, ax = plt.subplots(figsize=(9, 6))
    order = list(agg.keys())
    if baseline_name in agg:
        order.remove(baseline_name)
        order = [baseline_name] + order

    for variant in order:
        a = agg[variant]
        if not a["iters"]:
            continue
        it = np.asarray(a["iters"])
        mean = np.asarray(a["rmse_mean"], dtype=float)
        std = np.asarray(a["rmse_std"], dtype=float)
        lw = 2.5 if variant == baseline_name else 1.6
        z = 5 if variant == baseline_name else 3
        line = ax.plot(it, mean, label=variant, linewidth=lw, zorder=z)
        color = line[0].get_color()
        ax.fill_between(
            it, mean - std, mean + std, alpha=0.15, color=color, linewidth=0
            )

    ax.set_xlabel("ALS iteration")
    ax.set_ylabel("Train RMSE (mean ± std over folds)")
    ax.set_title("Convergence of ablations")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    _safe_savefig(fig, conv_dir / "convergence_all.png")


def _row_to_dict(
    r: AblationResultRow,
    feature_names: List[str]
) -> Dict[str, Any]:
    """Convert AblationResultRow to a flat dict for CSV/JSON.

    Args:
        r: AblationResultRow instance.
        feature_names: List of item feature names.

    Returns:
        Flat dict representation.
    """
    d: Dict[str, Any] = {
        "variant": r.variant,
        "rmse_mean": r.rmse_mean,
        "rmse_std": r.rmse_std,
        "time_mean": r.time_mean,
        "time_std": r.time_std,
        "mean_iters": r.mean_iters,
        "early_stopped_folds": r.early_stopped_folds,
        "target_n_iters": r.target_n_iters,
        "es_tol": r.es_tol,
        "es_min_iters": r.es_min_iters,
        "p_raw": r.p_raw,
        "p_fdr": r.p_fdr,
        "delta_mean": r.delta_mean,
    }
    # Per-bin RMSEs
    d.update(sorted(r.rmse_bins.items()))
    # Common hyperparameters for traceability
    for k in ("alpha", "graph_feature", "pop_reg_mode",
              "n_factors", "n_iters", "lambda_u", "lambda_v",
              "lambda_bu", "lambda_bi", "update_w_every"):
        if k in r.params:
            d[f"param_{k}"] = r.params[k]
    # λ_w_* for all features
    for f in feature_names:
        d[f"param_lambda_w_{f}"] = r.params.get(f"lambda_w_{f}")

    return d


def run_ablation(
    R_path: str,
    folds_path: str,
    best_params_path: str,
    features: Dict[str, np.ndarray],
    out_dir: str,
    n_pop_bins: int = N_POP_BINS,
    es_tol: Optional[float] = None,
    es_min_iters: Optional[int] = None,
    verbose_fit: int = 0,
) -> Tuple[Path, Path]:
    """Run ablations on frozen folds; write CSV + JSON with metrics & stats.

    Args:
        R_path: Path to ratings matrix `.npy` file.
        folds_path: Path to frozen folds `.npz` file.
        best_params_path: Path to best params JSON file.
        features: Dict of item side features.
        out_dir: Output directory to save artifacts.
        n_pop_bins: Number of popularity bins.
        es_tol: Early-stopping tolerance (overrides tuning default if set).
        es_min_iters: Minimum iterations before early stopping
                      (overrides tuning default if set).
        verbose_fit: Verbosity level for ALS fitting.

    Returns:
        Tuple of paths to the generated CSV and JSON files.
    """
    out_base = Path(out_dir) / "ablations"
    out_base.mkdir(parents=True, exist_ok=True)

    # Load data & folds
    R = np.load(R_path)
    folds, shape, saved_seed = load_folds_npz(folds_path)
    if tuple(shape) != tuple(R.shape):
        raise AssertionError("Folds were built for a different matrix shape.")

    # Load best params (accept {"params": {...}} or raw dict)
    best_payload = json.loads(Path(best_params_path).read_text())
    best_params = dict(
        best_payload["params"]
        ) if "params" in best_payload else dict(best_payload)

    # Early stopping defaults (aligned with tuning)
    es_tol = float(TUNE_ES_TOL if es_tol is None else es_tol)
    es_min_iters = int(
        TUNE_ES_MIN_ITERS if es_min_iters is None else es_min_iters
        )

    # Popularity binning
    item_bin, edges = _popularity_bins_from_R(
        R, n_bins=n_pop_bins, strategy=POP_BIN_STRATEGY
        )

    # Variants to evaluate
    feature_names = list(features.keys())
    variants = _variant_grid(best_params, feature_names)

    # Evaluate variants
    rows: List[AblationResultRow] = []
    fold_scores: Dict[str, List[float]] = {}
    curves: Dict[str, List[List[float]]] = {}

    for name, params in variants:
        fold_rmse, fold_time, fold_bin_rmse, fold_iters = _eval_variant_cv(
            variant_name=name,
            R=R,
            features=features,
            folds=folds,
            params=params,
            item_bin=item_bin,
            n_pop_bins=n_pop_bins,
            es_tol=es_tol,
            es_min_iters=es_min_iters,
            convergence_curves=curves,
            verbose_fit=verbose_fit,
        )
        fold_scores[name] = list(fold_rmse)

        target_n_iters = int(params.get("n_iters", 0))
        rows.append(AblationResultRow(
            variant=name,
            rmse_mean=float(np.mean(fold_rmse)),
            rmse_std=float(
                np.std(fold_rmse, ddof=1)
                ) if len(fold_rmse) > 1 else 0.0,
            time_mean=float(np.mean(fold_time)),
            time_std=float(
                np.std(fold_time, ddof=1)
                ) if len(fold_time) > 1 else 0.0,
            mean_iters=float(np.mean(fold_iters)),
            early_stopped_folds=int(
                sum(it < target_n_iters for it in fold_iters)
                ),
            target_n_iters=target_n_iters,
            es_tol=es_tol,
            es_min_iters=es_min_iters,
            rmse_bins=_aggregate_bins_mean(fold_bin_rmse),
            params=params,
        ))

    # Sign test vs baseline ("full"), with BH-FDR
    if "full" in fold_scores:
        base = fold_scores["full"]
        pvals = []
        for r in rows:
            if r.variant == "full":
                continue
            r.p_raw = _sign_test_paired(fold_scores[r.variant], base)
            r.delta_mean = float(
                np.mean(np.array(fold_scores[r.variant]) - np.array(base))
                )
            pvals.append(r.p_raw)
        if pvals:
            adj = _fdr_bh(pvals)
            j = 0
            for r in rows:
                if r.variant == "full":
                    continue
                r.p_fdr = float(adj[j])
                j += 1

    # Write CSV
    csv_path = out_base / "ablations.csv"
    df = pd.DataFrame([_row_to_dict(r, feature_names) for r in rows])
    df.to_csv(csv_path.as_posix(), index=False)

    # Write JSON (metadata + results)
    json_path = out_base / "ablations.json"
    payload = {
        "seed": DEFAULT_RANDOM_STATE,
        "matrix_shape": [int(R.shape[0]), int(R.shape[1])],
        "folds_seed": int(saved_seed),
        "feature_names": feature_names,
        "n_pop_bins": int(n_pop_bins),
        "pop_bin_edges": [float(e) for e in edges],
        "es_tol": es_tol,
        "es_min_iters": es_min_iters,
        "variants_evaluated": [r.variant for r in rows],
        "best_params_used": best_params,
        "results": [ _row_to_dict(r, feature_names) for r in rows ],
    }
    Path(json_path).write_text(json.dumps(payload, indent=2))

    # Plots (comparative + convergence)
    try:
        _save_comparative_plots(df, out_base)
    except Exception:
        pass
    try:
        _save_convergence_artifacts(curves, out_base, baseline_name="full")
    except Exception:
        pass

    return csv_path, json_path
