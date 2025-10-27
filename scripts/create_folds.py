"""
Entrywise K-fold splits for a ratings matrix.

This module creates disjoint validation folds over the observed entries of a
ratings matrix `R` (NaN = missing). It also provides utilities to save those
folds to a compressed `.npz` with metadata and load them back later, and a
helper to materialize train/valid splits for a given fold.


## Public API:

- `make_entrywise_folds(...)`  — build K disjoint folds over observed entries
- `save_folds_npz(...)`        — persist folds + metadata to a `.npz`
- `load_folds_npz(...)`        — read folds + metadata back from `.npz`
- `make_train_valid_split(...)`— construct `(R_train, R_valid, val_idx)`

## Typical usage:

```python
import numpy as np

from scripts.create_folds import (
    make_entrywise_folds,
    save_folds_npz,
    load_folds_npz,
    make_train_valid_split
)

R = np.load("data/ratings.npy")
folds_path = "artifacts/folds/entrywise_5_fold_seed_42.npz"

# Build folds once and save them
folds = make_entrywise_folds(R, n_splits=5, seed=42, shuffle=True)
save_folds_npz(folds_path, folds, R.shape, seed=42)

# Later, anywhere else in the project, load and use them
folds, shape, seed = load_folds_npz(folds_path)
R_train, R_valid, val_idx = make_train_valid_split(R, folds, k=0)
```
"""

from __future__ import annotations

from typing import List, Tuple
import os

import numpy as np


def make_entrywise_folds(
    R: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    shuffle: bool = True,
) -> List[np.ndarray]:
    """
    Build K disjoint validation splits over the observed entries of R.

    Args:
        R: (m x n) rating matrix with NaN for missing entries.
        n_splits: Number of folds K.
        seed: Random seed for shuffling.
        shuffle: Whether to shuffle observed entries before splitting.

    Returns:
        List of K arrays of flat indices into R used as validation entries.

    Example:
        folds = make_entrywise_folds(R=R, n_splits=5, seed=42)
        # folds[0] are the flat indices used as validation entries for fold 0.
    """
    # Set up random generator
    rng = np.random.default_rng(seed)

    # Get observed entries
    obs = np.flatnonzero(~np.isnan(R))

    # Shuffle if needed
    if shuffle:
        rng.shuffle(obs)

    # Split into K folds
    chunks = np.array_split(obs, n_splits)
    folds = [np.asarray(c, dtype=np.int64) for c in chunks]

    # Sanity checks
    assert sum(len(f) for f in folds) == len(obs)
    assert len(set(np.concatenate(folds).tolist())) == len(obs)

    return folds


def save_folds_npz(
    path: str,
    folds: List[np.ndarray],
    shape: Tuple[int, int],
    seed: int
) -> None:
    """
    Save folds and metadata to .npz for later loading.

    Args:
        path: Path to the .npz file.
        folds: List of K arrays of flat indices into R used as validation.
        shape: Shape of the original rating matrix R.
        seed: Random seed used for shuffling.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        shape=np.asarray(shape, dtype=np.int64),
        seed=np.asarray([seed], dtype=np.int64),
        **{f"fold{i}": f for i, f in enumerate(folds)}
    )


def load_folds_npz(
    path: str
) -> Tuple[List[np.ndarray], Tuple[int, int], int]:
    """
    Load folds and metadata from .npz file.
    
    Args:
        path: Path to the .npz file.

    Returns:
        Tuple of (folds, shape, seed) where:
            folds: List of K arrays of flat indices into R used as validation.
            shape: Shape of the original rating matrix R.
            seed: Random seed used for shuffling.
    """
    # Load data
    data = np.load(path, allow_pickle=False)

    # Extract metadata
    shape = tuple(int(x) for x in data["shape"])
    seed = int(data["seed"][0])

    # Fold keys are fold0, fold1, ...
    fold_keys = sorted([k for k in data.files if k.startswith("fold")],
                       key=lambda k: int(k.replace("fold", "")))

    # Extract folds
    folds = [data[k].astype(np.int64) for k in fold_keys]

    return folds, shape, seed


def matrix_from_indices(
    shape: Tuple[int, int],
    flat_idx: np.ndarray,
    flat_vals: np.ndarray
) -> np.ndarray:
    """
    Build a matrix of given shape from flat indices and values.

    Args:
        shape: Desired shape of the output matrix.
        flat_idx: 1D array of flat indices into the matrix.
        flat_vals: 1D array of values corresponding to flat_idx.

    Returns:
        Matrix of given shape with values at specified indices and NaN elsewhere.
    """
    # Initialize with NaNs
    M = np.full(shape[0] * shape[1], np.nan, dtype=float)

    # Assign values at specified indices
    M[flat_idx] = flat_vals

    return M.reshape(shape)


def make_train_valid_split(
    R: np.ndarray,
    folds: List[np.ndarray],
    k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given rating matrix R and folds, build train and val matrices for fold k.

    Args:
        R: (m x n) rating matrix with NaN for missing entries.
        folds: List of K arrays of flat indices into R used as validation.
        k: Fold index to use for validation.

    Returns:
        Tuple of (R_train, R_valid, val_idx) where:
            R_train: Training rating matrix for fold k.
            R_valid: Validation rating matrix for fold k.
            val_idx: Flat indices used as validation entries for fold k.
    """
    # Extract shape
    m, n = R.shape

    # Get train and valid indices
    all_obs = np.flatnonzero(~np.isnan(R))
    val_idx = folds[k]
    train_idx = np.setdiff1d(all_obs, val_idx, assume_unique=False)

    # Build train and validation matrices
    R_train = matrix_from_indices((m, n), train_idx, R.ravel()[train_idx])
    R_val = matrix_from_indices((m, n), val_idx, R.ravel()[val_idx])

    return R_train, R_val, val_idx
