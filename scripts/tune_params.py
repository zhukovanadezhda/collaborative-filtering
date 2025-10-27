"""
ALS hyperparameter tuning with Optuna on frozen CV folds.

This module tunes an ALS MF recommender by running an Optuna study over a
defined search space. To change the search space, modify the the global
constants in the beginning of the file. The module supports early stopping,
pruning unpromising trials, and checkpointing intermediate results. It is
supposed to be run on precomputed folds saved in .npz format. If you don't have
such folds, use scripts/create_folds.py to create them.

Public API:
----------
- run_tuning(...)         : runs the study and writes artifacts.
- make_checkpoint_cb(...) : periodic saver for long studies.
- save_all_artifacts(...) : exports CSV/plots/JSON from a finished study.

Results (under: <out_dir>/tuning):
----------
  ├─ <study_name>_trials.csv         # Trials with params, value, and metadata
  ├─ <study_name>_summary.json       # Study summary + best trial metadata
  ├─ <study_name>_best_params.json   # Best trial params + value + user_attrs
  └─ plots/
       ├─ history.html               # Optimization history
       ├─ intermediate_values.html   # Per-trial intermediate fold RMSEs
       ├─ param_importances.html     # (If enough trials) Param importances
       ├─ slice.html                 # Slices for key parameters
       ├─ parallel_coordinates.html  # Parallel coords for key parameters
       └─ contour_*.html             # A few pairwise contours for key params

Typical usage
-------------
from scripts.tune_params import run_tuning

R_PATH = "data/ratings.npy"
FOLDS_PATH = "artifacts/folds/entrywise_5_fold_seed_42.npz"

# Example side features (item-level)
genres = np.load("data/genres.npy")
years  = np.load("data/years.npy").reshape(-1, 1)
features = {"genres": genres, "years": years}

res = run_tuning(
    R_path=R_PATH,
    folds_path=FOLDS_PATH,
    features=features,
    out_dir="results",
    study_name="als_tune_v1",
    n_trials=150,
    timeout_sec=None,
    seed=42,
    save_every=25,
    verbose_fit=0
)

print("Best value:", res.best_value)
print("Best params JSON:", res.best_params_json_path)
print("Trials CSV:", res.trials_csv_path)
print("Plots dir:", res.plots_dir)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_intermediate_values,
    plot_param_importances,
    plot_slice,
    plot_parallel_coordinate,
    plot_contour,
)

from scripts.als import ALS
from scripts.als_config import (
    ALSConfig,
    CoreConfig,
    BiasesConfig,
    GraphConfig,
    GraphSimConfig,
)
from scripts.create_folds import load_folds_npz, make_train_valid_split


# Search space bounds
N_FACTORS_MIN: int = 1
N_FACTORS_MAX: int = 150
N_ITERS_MIN: int = 100
N_ITERS_MAX: int = 100
LAMBDA_MIN: float = 1e-4
LAMBDA_MAX: float = 1e4
UPDATE_W_EVERY_MIN: int = 1
UPDATE_W_EVERY_MAX: int = 60
ALPHA_MIN: float = 0.0
ALPHA_MAX: float = 100.0
S_TOPK_MIN: int = 1
S_TOPK_MAX: int = 610
S_EPS_MIN: float = 1e-10
S_EPS_MAX: float = 1e-4

# Early stopping defaults (set to None to disable)
ES_TOL: float = 1e-4
ES_MIN_ITERS: int = 10

# Plots
MAX_CONTOUR_PAIRS: int = 6

# Misc
DEFAULT_RANDOM_STATE: int = 42


@dataclass
class TuningResult:
    """Container with the final study result and artifact locations."""
    study_name: str
    best_value: float
    best_params: Dict[str, Any]
    n_trials: int
    n_complete: int
    n_pruned: int
    artifacts_dir: str
    trials_csv_path: str
    summary_json_path: str
    best_params_json_path: str
    plots_dir: str


def _assert_finite_features(features: Dict[str, np.ndarray]) -> None:
    """Raise if any feature contains non-finite values."""
    for name, X in features.items():
        if not np.isfinite(X).all():
            raise ValueError(f"Feature '{name}' has non-finite values.")


def _rmse_on_indices(
    R_true: np.ndarray,
    R_pred: np.ndarray,
    flat_idx: np.ndarray
) -> float:
    """Compute RMSE only over the provided flat indices.

    Args:
        R_true: Ground truth ratings matrix (with NaN for missing).
        R_pred: Predicted ratings matrix.
        flat_idx: 1D array of flat indices into R_true/R_pred to evaluate.

    Returns:
        RMSE value (float).

    """
    if flat_idx.size == 0:
        return float("nan")
    y_true = R_true.ravel()[flat_idx]
    y_pred = R_pred.ravel()[flat_idx]
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _search_space(
    trial: optuna.Trial,
    feature_names: List[str]
) -> Dict[str, Any]:
    """Define the Optuna search space.
    
    Args:
        trial: Optuna trial object.
        feature_names: List of available feature names for graph construction.

    Returns:
        A dictionary with the suggested hyperparameters.
    """

    # Determine if graph features are available
    graph_choices = feature_names if feature_names else ["__none__"]

    params: Dict[str, Any] = {
        # Core
        "n_factors": trial.suggest_int(
            "n_factors", N_FACTORS_MIN, N_FACTORS_MAX
            ),
        "n_iters": trial.suggest_int(
            "n_iters", N_ITERS_MIN, N_ITERS_MAX
            ),
        "lambda_u": trial.suggest_float(
            "lambda_u", LAMBDA_MIN, LAMBDA_MAX, log=True
            ),
        "lambda_v": trial.suggest_float(
            "lambda_v", LAMBDA_MIN, LAMBDA_MAX, log=True
        ),
        "lambda_bu": trial.suggest_float(
            "lambda_bu", LAMBDA_MIN, LAMBDA_MAX, log=True
        ),
        "lambda_bi": trial.suggest_float(
            "lambda_bi", LAMBDA_MIN, LAMBDA_MAX, log=True
        ),
        "pop_reg_mode": trial.suggest_categorical(
            "pop_reg_mode", [None, "inverse_sqrt"]
        ),
        "update_w_every": trial.suggest_int(
            "update_w_every", UPDATE_W_EVERY_MIN, UPDATE_W_EVERY_MAX
        ),
        # Graph
        "alpha": trial.suggest_float(
            "alpha", ALPHA_MIN, ALPHA_MAX
        ),
        "S_topk": trial.suggest_int(
            "S_topk", S_TOPK_MIN, S_TOPK_MAX
        ),
        "S_eps": trial.suggest_float(
            "S_eps", S_EPS_MIN, S_EPS_MAX, log=True
        ),
        "graph_feature": trial.suggest_categorical(
            "graph_feature", graph_choices
        ),
    }

    # Add λ_w_<feature> for every provided feature
    for name in feature_names:
        params[f"lambda_w_{name}"] = trial.suggest_float(
            f"lambda_w_{name}", LAMBDA_MIN, LAMBDA_MAX, log=True
        )

    return params


def _normalize_params(
    params: Dict[str, Any],
    R_shape: tuple[int, int],
    feature_names: List[str]
    ) -> Dict[str, Any]:
    """Clip parameters to be consistent with data shapes and available features.

    Args:
        params: Dictionary of trial parameters.
        R_shape: Shape of the ratings matrix (n_users, n_items).
        feature_names: List of available feature names for graph construction.

    Returns:
        Normalized parameters dictionary.
    """
    # Unpack shape
    m, n = R_shape

    # Clip latent dimensionality to matrix rank upper bound
    params["n_factors"] = max(
        N_FACTORS_MIN,
        min(int(params["n_factors"]), min(m, n))
    )

    # Clip S_topk to at most n-1
    params["S_topk"] = max(
        S_TOPK_MIN,
        min(int(params["S_topk"]), max(1, n - 1))
    )

    # Ensure update_w_every ≤ n_iters
    params["update_w_every"] = max(
        UPDATE_W_EVERY_MIN,
        min(int(params["update_w_every"]), int(params["n_iters"]))
    )

    # Disable graph if no matching feature
    if (not feature_names) or (params.get("graph_feature") not in feature_names):
        params["alpha"] = 0.0
        params["graph_feature"] = "__none__"

    return params


def _make_config(params: Dict[str, Any]) -> ALSConfig:
    """Build the ALSConfig from trial parameters.

    Args:
        params: Dictionary of trial parameters.

    Returns:
        ALSConfig object.
    """
    core = CoreConfig(
        n_factors=int(params["n_factors"]),
        n_iters=int(params["n_iters"]),
        lambda_u=float(params["lambda_u"]),
        lambda_v=float(params["lambda_v"]),
        pop_reg_mode=params.get("pop_reg_mode", None),
        random_state=DEFAULT_RANDOM_STATE,
        update_w_every=int(params.get("update_w_every", UPDATE_W_EVERY_MIN)),
    )

    biases = BiasesConfig(
        lambda_bu=float(params.get("lambda_bu", core.lambda_u)),
        lambda_bi=float(params.get("lambda_bi", core.lambda_v)),
    )

    # Build graph config
    alpha = float(params.get("alpha", 0.0))
    # Determine if graph is enabled
    gfeat = params.get("graph_feature", "__none__")
    # Disable graph if alpha ≤ 0 or no feature selected
    if alpha <= 0.0 or gfeat == "__none__":
        graph = GraphConfig(alpha=0.0, sim=None)
    else:
        sim = GraphSimConfig(
            source="feature",
            feature_name=gfeat,
            metric="cosine",
            topk=int(params.get("S_topk", 50)),
            eps=float(params.get("S_eps", 1e-8)),
        )
        graph = GraphConfig(alpha=alpha, sim=sim)

    return ALSConfig(core=core, biases=biases, graph=graph)


def _params_to_lambda_w(
    params: Dict[str, Any],
    features: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
    """Map parameters to per-feature λ_w values.

    Args:
        params: Dictionary of trial parameters.
        features: Dictionary of feature matrices.

    Returns:
        Dictionary mapping feature names to their corresponding λ_w values.
    """
    out: Dict[str, float] = {}
    for name in features.keys():
        out[name] = float(params.get(f"lambda_w_{name}", 0.0))

    return out


def _cv_score_single_trial(
    params: Dict[str, Any],
    R: np.ndarray,
    features: Dict[str, np.ndarray],
    folds: List[np.ndarray],
    verbose_fit: int = 0,
    trial: Optional[optuna.Trial] = None,
) -> float:
    """Compute mean RMSE across folds; report to Optuna and support pruning.
    
    Args:
        params: Dictionary of trial parameters.
        R: User-item interaction matrix.
        features: Dictionary of feature matrices.
        folds: List of cross-validation folds.
        verbose_fit: Verbosity level for fitting.
        trial: Optuna trial object for reporting.

    Returns:
        Mean RMSE across folds.
    """
    cfg = _make_config(params)
    lambda_w = _params_to_lambda_w(params, features)

    fold_scores: List[float] = []
    iters_per_fold: List[int] = []
    early_flags: List[bool] = []

    target_iters = int(cfg.core.n_iters)

    for i, _ in enumerate(folds):
        # Build fold matrices (train uses all except current fold)
        R_train, R_valid, val_idx = make_train_valid_split(R, folds, i)

        # Fit ALS with early stopping
        model = ALS(config=cfg, lambda_w=lambda_w)
        model.fit(
            R_train,
            features=features,
            tol=ES_TOL,
            min_iters=ES_MIN_ITERS,
            verbose=verbose_fit,
        )

        # Track iterations executed and early stop flag
        n_run = len(model.history.get("train_rmse", []))
        iters_per_fold.append(n_run)
        early_flags.append(n_run < target_iters)

        # Predict full matrix and evaluate only on validation indices
        R_hat = model.predict(features=features)
        rmse = _rmse_on_indices(R_valid, R_hat, val_idx)
        fold_scores.append(rmse)

        # Report intermediate score to Optuna and allow pruning
        if trial is not None:
            trial.report(rmse, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Optional console feedback
        if verbose_fit > 0:
            print(
                f"  Fold {i + 1}/{len(folds)} rmse={rmse:.5f} iters={n_run} "
                f"{'(early stop)' if early_flags[-1] else ''}",
                flush=True,
            )

    mean_rmse = float(np.mean(fold_scores))

    # Persist ES metadata in user_attrs so it lands in CSV/JSON artifacts
    if trial is not None:
        trial.set_user_attr("es_tol", float(ES_TOL))
        trial.set_user_attr("es_min_iters", int(ES_MIN_ITERS))
        trial.set_user_attr("target_n_iters", target_iters)
        trial.set_user_attr("iters_per_fold", iters_per_fold)
        trial.set_user_attr("mean_iters", float(np.mean(iters_per_fold)))
        trial.set_user_attr("early_stopped_folds", int(sum(early_flags)))
        trial.set_user_attr("fold_rmse", fold_scores)

    return mean_rmse


def _safe_plot(callable_plot, path: Path) -> None:
    """Safely render and save an Optuna plot to HTML file.

    Args:
        callable_plot: The Optuna plot to render.
        path: The file path to save the plot.
    """
    try:
        callable_plot.write_html(path.as_posix())
    except Exception:
        # Plotting can fail for few trials; ignoring is acceptable
        pass


def save_all_artifacts(
    study: optuna.Study,
    out_dir: Path,
    study_name: str,
    params_subset: Optional[List[str]] = None,
    max_pairs: int = MAX_CONTOUR_PAIRS,
    summary_ctx: Optional[Dict[str, Any]] = None,
) -> None:
    """Save trials CSV, plots, and JSON summaries for the study.

    Args:
        study: The Optuna study object.
        out_dir: The output directory for saving artifacts.
        study_name: The name of the study.
        params_subset: Optional list of parameter names to include in plots.
        max_pairs: Maximum number of pairwise contour plots to generate.
        summary_ctx: Optional context dictionary for summary information.
    """
    # Prepare output directories
    tuning_dir = out_dir / "tuning"
    plots_dir = tuning_dir / "plots"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save trials.csv with expanded user_attrs
    df = study.trials_dataframe()

    # Expand simple user attributes
    def _col(k): return [t.user_attrs.get(k) for t in study.trials]

    for k in (
        "mean_iters",
        "early_stopped_folds",
        "target_n_iters",
        "es_tol",
        "es_min_iters"
        ):
        df[f"user_{k}"] = _col(k)

    # Lists → JSON strings for CSV stability
    df["user_iters_per_fold"] = [json.dumps(
        t.user_attrs.get("iters_per_fold")
        ) for t in study.trials]
    df["user_fold_rmse"] = [json.dumps(
        t.user_attrs.get("fold_rmse")
        ) for t in study.trials]

    trials_csv = tuning_dir / f"{study_name}_trials.csv"
    df.to_csv(trials_csv.as_posix(), index=False)

    # Determine params_subset for multi-parameter plots
    if params_subset is None and study.best_trial is not None:
        params_subset = list(study.best_trial.params.keys())

    # Save plots (HTML)
    _safe_plot(
        plot_optimization_history(study),
        plots_dir / "history.html"
        )
    _safe_plot(
        plot_intermediate_values(study),
        plots_dir / "intermediate_values.html"
        )
    _safe_plot(
        plot_param_importances(study),
        plots_dir / "param_importances.html"
    )
    if params_subset:
        _safe_plot(
            plot_slice(study, params=params_subset),
            plots_dir / "slice.html"
        )
        _safe_plot(
            plot_parallel_coordinate(study, params=params_subset),
            plots_dir / "parallel_coordinates.html",
        )
        # Save a few pairwise contour plots
        for i, (a, b) in enumerate(combinations(params_subset, 2), start=1):
            if i > max_pairs:
                break
            _safe_plot(
                plot_contour(study, params=[a, b]),
                plots_dir / f"contour_{i}_{a}_vs_{b}.html",
            )

    # Save summary.json
    best_trial = study.best_trial
    payload_summary = {
        "best_value": float(study.best_value),
        "best_trial": int(best_trial.number),
        "n_complete": int(sum(t.state.name == "COMPLETE" for t in study.trials)),
        "n_pruned": int(sum(t.state.name == "PRUNED" for t in study.trials)),
        "user_attrs": dict(best_trial.user_attrs),        # Contains ES metadata
    }
    if summary_ctx:
        payload_summary.update(summary_ctx)
    with open(tuning_dir / f"{study_name}_summary.json", "w") as f:
        json.dump(payload_summary, f, indent=2)

    # Save best_params.json
    payload_best = {
        "best_value": float(best_trial.value),
        "best_trial": int(best_trial.number),
        "params": dict(best_trial.params),
        "user_attrs": dict(best_trial.user_attrs),
    }
    with open(tuning_dir / f"{study_name}_best_params.json", "w") as f:
        json.dump(payload_best, f, indent=2)


def make_checkpoint_cb(
    study: optuna.Study,
    out_dir: Path,
    study_name: str,
    save_every: int,
    expected_n_trials: Optional[int] = None,
    save_on_last: bool = True,
    params_subset: Optional[List[str]] = None,
    max_pairs: int = MAX_CONTOUR_PAIRS,
    summary_ctx: Optional[Dict[str, Any]] = None,
) -> optuna.study.StudyCallback:
    """Build a callback that saves artifacts every `save_every` trials.

    Args:
        study: The Optuna study object.
        out_dir: The output directory for saving artifacts.
        study_name: The name of the study.
        save_every: Save artifacts every N trials.
        expected_n_trials: Expected total number of trials.
        save_on_last: Whether to save artifacts on the last trial.
        params_subset: Optional list of parameter names to include in plots.
        max_pairs: Maximum number of pairwise contour plots to generate.
        summary_ctx: Optional context dictionary for summary information.

    Returns:
        An Optuna study callback function.
    """
    def cb(_study: optuna.Study, trial: optuna.trial.FrozenTrial):
        # Determine whether to save: periodic or final
        num = trial.number + 1
        periodic = save_every > 0 and (num % save_every == 0)
        is_last = (
            save_on_last and 
            expected_n_trials is not None and 
            num == expected_n_trials
            )
        if periodic or is_last:
            try:
                save_all_artifacts(
                    study=_study,
                    out_dir=out_dir,
                    study_name=study_name,
                    params_subset=params_subset,
                    max_pairs=max_pairs,
                    summary_ctx=summary_ctx,
                )
            except Exception:
                # Non-fatal; continue tuning
                pass
    return cb


def run_tuning(
    R_path: str,
    folds_path: str,
    features: Dict[str, np.ndarray],
    out_dir: str,
    study_name: str = "als_tuning",
    n_trials: int = 50,
    timeout_sec: Optional[int] = None,
    seed: int = DEFAULT_RANDOM_STATE,
    save_every: int = 50,
    verbose_fit: int = 0,
) -> TuningResult:
    """Run TPE + MedianPruner on frozen folds and write artifacts periodically.

    Args:
        R_path: Path to the ratings matrix (.npy).
        folds_path: Path to the folds (.npz).
        features: Dictionary of feature matrices.
        out_dir: Output directory for artifacts.
        study_name: Name of the Optuna study.
        n_trials: Number of trials to run.
        timeout_sec: Optional timeout in seconds.
        seed: Random seed for reproducibility.
        save_every: Save artifacts every N trials.
        verbose_fit: Verbosity level for fitting.

    Returns:
        TuningResult object with study summary and artifact paths.
    """
    # Validate features
    _assert_finite_features(features)

    # Prepare IO paths
    out_base = Path(out_dir)
    tuning_dir = out_base / "tuning"
    plots_dir = tuning_dir / "plots"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data and folds
    R = np.load(R_path)
    folds, shape, saved_seed = load_folds_npz(folds_path)
    if tuple(shape) != tuple(R.shape):
        raise AssertionError("Folds were built for a different matrix shape.")

    # Build study (TPE + median pruner)
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=min(5, max(2, n_trials // 6)))
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    feature_names = list(features.keys())

    def objective(trial: optuna.Trial) -> float:
        # Sample and normalize parameters for this trial
        params = _search_space(trial, feature_names)
        params = _normalize_params(params, R.shape, feature_names)

        # Evaluate by CV with pruning + early stopping inside ALS
        return _cv_score_single_trial(
            params=params,
            R=R,
            features=features,
            folds=folds,
            verbose_fit=verbose_fit,
            trial=trial,
        )

    # Checkpoint context for summary.json
    summary_ctx = {
        "seed": seed,
        "folds_seed": int(saved_seed),
        "matrix_shape": [int(R.shape[0]), int(R.shape[1])],
        "feature_names": feature_names,
        "es_tol": ES_TOL,
        "es_min_iters": ES_MIN_ITERS,
    }

    # Build periodic/last-trial saver
    ckpt_cb = make_checkpoint_cb(
        study=study,
        out_dir=out_base,
        study_name=study_name,
        save_every=save_every,
        expected_n_trials=n_trials,
        save_on_last=True,
        params_subset=None,
        max_pairs=MAX_CONTOUR_PAIRS,
        summary_ctx=summary_ctx,
    )

    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        show_progress_bar=True,
        callbacks=[ckpt_cb],
    )

    # Final save to capture the last state (covers timeout, pruning end, etc.)
    save_all_artifacts(
        study=study,
        out_dir=out_base,
        study_name=study_name,
        params_subset=list(study.best_trial.params.keys()),
        max_pairs=MAX_CONTOUR_PAIRS,
        summary_ctx=summary_ctx,
    )

    # Compose return object with paths
    trials_csv_path = (
        tuning_dir / f"{study_name}_trials.csv"
    ).as_posix()
    summary_json_path = (
        tuning_dir / f"{study_name}_summary.json"
    ).as_posix()
    best_params_json_path = (
        tuning_dir / f"{study_name}_best_params.json"
    ).as_posix()

    return TuningResult(
        study_name            = study.study_name,
        best_value            = float(study.best_value),
        best_params           = dict(study.best_trial.params),
        n_trials              = len(study.trials),
        n_complete            = sum(
            t.state.name == "COMPLETE" for t in study.trials
            ),
        n_pruned              = sum(
            t.state.name == "PRUNED" for t in study.trials
            ),
        artifacts_dir         = tuning_dir.as_posix(),
        trials_csv_path       = trials_csv_path,
        summary_json_path     = summary_json_path,
        best_params_json_path = best_params_json_path,
        plots_dir             = plots_dir.as_posix(),
    )
