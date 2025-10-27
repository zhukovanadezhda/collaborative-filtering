"""
Feature normalization module.

This module standardizes item feature matrices before feeding them into ALS.
It supports optional median imputation and several row/column normalization
modes. If imputation is not requested, the presence of NaN/Inf values raises
an error. Columns and rows that are constant (zero variance/range) are clipped
using a small epsilon constant to avoid division by zero.

## Public API:

- normalize_feature(...)        : normalize a single (n_items x d) array.
- normalize_features_dict(...)  : normalize a {name -> array} dict with shared
                                  defaults and optional per-feature overrides.

## Typical usage

```
from prepare_features import normalize_feature, normalize_features_dict
import numpy as np

# Example raw features
genres = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [1, 0, 1]], dtype=float)
years  = np.array([1995, 2002, 2010], dtype=float)

# Normalize individually
G = normalize_feature(genres, method="row_l2",     impute="col_median")
Y = normalize_feature(years,  method="col_zscore", impute="col_median")

# Or normalize a dict with shared defaults + overrides
features = normalize_features_dict(
    {"genres": genres, "years": years},
    method="none",            # default for all
    impute="col_median",
    per_feature_overrides={
        "genres": {"method": "row_l2"},
        "years":  {"method": "col_zscore"},
    },
)
```
"""

from __future__ import annotations

from typing import Dict, Any, Mapping, Literal
import numpy as np

# Global constants
DEFAULT_DTYPE = "float32"
DEFAULT_EPS = 1e-8


def _as_2d(X: np.ndarray) -> np.ndarray:
    """Ensure shape (n_items, d)."""
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _copy_if(X: np.ndarray, copy: bool) -> np.ndarray:
    """Copy X if requested."""
    return X.copy() if copy else X


def _replace_nonfinite_with_nan(X: np.ndarray) -> None:
    """In-place: set ±inf to NaN (used before imputation)."""
    mask = ~np.isfinite(X)
    if mask.any():
        X[mask] = np.nan


def _validate_finite(X: np.ndarray) -> None:
    """Raise ValueError if X contains NaN/Inf."""
    if not np.isfinite(X).all():
        raise ValueError("Input feature contains NaN/Inf and impute='none'.")


def _cast(X: np.ndarray, dtype: str) -> np.ndarray:
    """Cast X to desired dtype."""
    return X.astype(dtype, copy=False)


def _impute_col_median_inplace(X: np.ndarray) -> None:
    """In-place column-median imputation for NaN/Inf values."""
    _replace_nonfinite_with_nan(X)
    if not np.isnan(X).any():
        return
    med = np.nanmedian(X, axis=0, keepdims=True)
    # if a whole column is NaN, fall back to zeros
    med = np.where(np.isfinite(med), med, 0.0)
    nan_rows, nan_cols = np.where(np.isnan(X))
    if nan_rows.size > 0:
        X[nan_rows, nan_cols] = med[0, nan_cols]


def _row_l1(X: np.ndarray, eps: float) -> np.ndarray:
    """Row-wise L1 normalization."""
    s = np.sum(np.abs(X), axis=1, keepdims=True)
    s = np.maximum(s, eps)
    return X / s


def _row_l2(X: np.ndarray, eps: float) -> np.ndarray:
    """Row-wise L2 normalization."""
    n = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    n = np.maximum(n, eps)
    return X / n


def _col_zscore(X: np.ndarray, eps: float) -> np.ndarray:
    """Column-wise z-score normalization."""
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    Xn = (X - mu) / sd
    # Safety for odd numerics
    Xn[~np.isfinite(Xn)] = 0.0
    return Xn


def _col_minmax(X: np.ndarray, eps: float) -> np.ndarray:
    """Column-wise min-max normalization to [0, 1]."""
    mn = np.min(X, axis=0, keepdims=True)
    mx = np.max(X, axis=0, keepdims=True)
    rng = np.maximum(mx - mn, eps)
    return (X - mn) / rng


def normalize_feature(
    X: np.ndarray,
    method: str = "none",
    *,
    impute: str = "none",
    eps: float = DEFAULT_EPS,
    dtype: str = DEFAULT_DTYPE,
    copy: bool = True,
) -> np.ndarray:
    """
    Normalize a feature matrix (n_items x d) with optional median imputation.

    Args:
        X : Feature matrix with one row per item. Can be (n,) or (n, d).
            If (n,), it is reshaped to (n, 1).
        method:
            - "none"       : return X after optional imputation + type cast
            - "row_l1"     : make each row L1-normalized (sum = 1)
            - "row_l2"     : make each row L2-normalized (norm = 1)
            - "col_zscore" : z-score each column (x - mean) / std
            - "col_minmax" : scale each column to [0, 1]
        impute:
            - "none"       : do not impute; NaN/Inf raises ValueError
            - "col_median" : replace NaN/Inf by column medians before normalize
        eps: Numerical stability constant (avoids division by zero).
        dtype: Output dtype (default: "float32").
        copy: If True, operate on a copy of X.

    Returns:
            Normalized feature matrix with dtype `dtype`.
    """

    # Raise error if uncorrect method or impute
    if method not in {"none", "row_l1", "row_l2", "col_zscore", "col_minmax"}:
        raise ValueError(f"Unknown method '{method}'.")
    if impute not in {"none", "col_median"}:
        raise ValueError(f"Unknown impute '{impute}'.")

    # Prepare input
    X = _as_2d(X)
    X = _copy_if(X, copy)

    # Impute or validate
    if impute == "col_median":
        _impute_col_median_inplace(X)
    else:
        _validate_finite(X)

    # Dispatch normalization
    if method == "none":
        return _cast(X, dtype)
    if method == "row_l1":
        return _cast(_row_l1(X, eps), dtype)
    if method == "row_l2":
        return _cast(_row_l2(X, eps), dtype)
    if method == "col_zscore":
        return _cast(_col_zscore(X, eps), dtype)
    if method == "col_minmax":
        return _cast(_col_minmax(X, eps), dtype)

    raise ValueError(f"Unknown method '{method}'.")


def normalize_features_dict(
    features: Mapping[str, np.ndarray],
    *,
    method: str = "none",
    impute: str = "none",
    eps: float = DEFAULT_EPS,
    dtype: str = DEFAULT_DTYPE,
    copy: bool = True,
    per_feature_overrides: Mapping[str, Mapping[str, Any]] | None = None
) -> Dict[str, np.ndarray]:
    """
    Normalize a dict of features with shared defaults and optional overrides.

    Args:
        features : Dictionary of raw features {name -> array}.
        method : Default normalization method for all features.
        impute : Default imputation method for all features.
        eps : Numerical stability constant.
        dtype : Output dtype for all features.
        copy : Whether to copy each feature before processing.
        per_feature_overrides : Per-feature kwargs.

    Returns:
        New dict with normalized features.
    """
    out: Dict[str, np.ndarray] = {}
    per_feature_overrides = per_feature_overrides or {}

    for name, X in features.items():
        overrides = dict(per_feature_overrides.get(name, {}))
        out[name] = normalize_feature(
            X,
            method=overrides.pop("method", method),
            impute=overrides.pop("impute", impute),
            eps=overrides.pop("eps", eps),
            dtype=overrides.pop("dtype", dtype),
            copy=overrides.pop("copy", copy),
            **overrides,  # allow future-proof extra kwargs
        )
    return out
