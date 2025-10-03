#!/usr/bin/env python3
"""
Alternating Least Squares (ALS) for collaborative filtering.

Implements matrix factorization with regularized least squares to impute
missing ratings. 
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from tqdm import trange

# Scale factor for random initialization of latent factors
SCALE_FACTOR = 0.1

# Set up logging
logging.basicConfig(
    # Default level = INFO; change to DEBUG for more detail
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ALS:
    """
    Alternating Least Squares (ALS) for collaborative filtering.

    Factorizes a user-item rating matrix R into two low-rank matrices:
        R ≈ U @ V.T
    where:
        - U: user_factors (m x k)
        - V: item_factors (n x k)
    """

    def __init__(
        self,
        n_factors: int,
        n_iters: int,
        reg: float,
        random_state: int | None = None,
    ) -> None:
        """
        Args:
            n_factors: Dimensionality of latent factor space (rank k).
            n_iters: Number of ALS iterations (alternating updates).
            reg: Regularization strength (ridge penalty λ).
            random_state: Optional random seed for reproducibility.
        """
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.reg = reg
        self.random_state = random_state

        # These will be initialized during fit()
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, R: np.ndarray) -> ALS:
        """
        Fit ALS model to rating matrix.

        Args:
            R: Ratings matrix with np.nan for missing entries.

        Returns:
            Self (allows chaining, e.g. model.fit(R).predict()).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # m users × n items
        m, n = R.shape

        # Random initialization of user/item factors with small Gaussian noise
        self.user_factors = np.random.normal(
            scale=SCALE_FACTOR,
            size=(m, self.n_factors)
            )
        self.item_factors = np.random.normal(
            scale=scale=SCALE_FACTOR,
            size=(n, self.n_factors)
            )

        # Mask = which entries in R are observed
        # (True = observed, False = missing)
        mask = ~np.isnan(R)

        logger.info(
            "Starting ALS training: m=%d users, n=%d items, "
            "factors=%d, n_iters=%d, reg=%.4f, random_state=%s",
            m, n, self.n_factors, self.n_iters, self.reg, self.random_state
        )

        # Main ALS loop
        for it in trange(self.n_iters, desc="ALS iterations"):
            # Update user factors
            for u in range(m):
                # Items that user u has rated
                idx = mask[u]
                # Skip users with no ratings
                if not np.any(idx):
                    continue

                V = self.item_factors[idx]  # Factors of rated items
                r = R[u, idx]               # Ratings given by user u

                # Solve normal equation: (VᵀV + λI) * U[u] = Vᵀr
                A = V.T @ V + self.reg * np.eye(self.n_factors)
                b = V.T @ r
                self.user_factors[u] = np.linalg.solve(A, b)

            # Update item factors
            for i in range(n):
                # Users that rated item i
                idx = mask[:, i]
                # Skip items with no ratings
                if not np.any(idx):
                    continue

                U = self.user_factors[idx]  # Factors of users who rated i
                r = R[idx, i]               # Ratings for item i

                # Solve normal equation: (UᵀU + λI) * V[i] = Uᵀr
                A = U.T @ U + self.reg * np.eye(self.n_factors)
                b = U.T @ r
                self.item_factors[i] = np.linalg.solve(A, b)

            logger.debug("Completed iteration %d", it + 1)

        logger.info("ALS training finished.")
        return self

    def predict(self) -> np.ndarray:
        """
        Predict the full rating matrix R̂ = U @ Vᵀ.

        Returns:
            Completed ratings matrix (np.ndarray).
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before prediction.")

        return self.user_factors @ self.item_factors.T


def compute_rmse(R_true: np.ndarray, R_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE) between true and predicted ratings.
    Missing entries (NaN) are ignored.

    Args:
        R_true: Ground truth ratings matrix (with NaN for missing).
        R_pred: Predicted ratings matrix.

    Returns:
        RMSE value (float).
    """
    # Only consider observed entries
    mask = ~np.isnan(R_true)
    if not np.any(mask):
        raise ValueError("No observed ratings in R_true.")

    return np.sqrt(np.mean((R_true[mask] - R_pred[mask]) ** 2))


def read_data(file_path: str) -> np.ndarray:
    """
    Load rating matrix from `.npy` file.

    Args:
        file_path: Path to .npy file containing np.ndarray.

    Returns:
        Rating matrix (np.ndarray).
    """
    return np.load(file_path)


def merge_train_test(R_train: np.ndarray, R_test: np.ndarray) -> np.ndarray:
    """
    Merge training and test matrices into a single ratings matrix.

    Args:
        R_train: Training ratings matrix (with NaN for missing).
        R_test: Test ratings matrix (with NaN for missing).

    Returns:
        Combined ratings matrix.
    """
    R_merged = R_train.copy()
    mask = ~np.isnan(R_test)
    R_merged[mask] = R_test[mask]

    return R_merged


def complete_ratings(
    train_path: str,
    test_path: str,
    params: Dict[str, int | float],
) -> np.ndarray:
    """
    High-level helper function to complete ratings.

    Loads data, merges train/test, trains ALS, and returns predictions.

    Args:
        train_path: Path to training `.npy` file.
        test_path: Path to test `.npy` file.
        params: Model hyperparameters (dict with keys):
            - n_factors
            - n_iters
            - reg
            - random_state

    Returns:
        Completed ratings matrix (np.ndarray).
    """
    # Load train and test sets
    R_train = read_data(train_path)
    R_test = read_data(test_path)

    # Merge them into a single observed matrix
    R_merged = merge_train_test(R_train, R_test)

    # Initialize and fit ALS model
    model = ALS(
        n_factors=int(params.get("n_factors")),
        n_iters=int(params.get("n_iters")),
        reg=float(params.get("reg")),
        random_state=params.get("random_state"),
    )

    model.fit(R_merged)

    return model.predict()
