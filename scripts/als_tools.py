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

from scripts.laplacian_tools import build_item_similarity_from_genres

# Scale factor for random initialization of latent factors
SCALE_FACTOR = 0.1

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


class ALS:
    """
    Alternating Least Squares (ALS) for collaborative filtering.

    Factorizes a user-item rating matrix R into three low-rank matrices:
    
        R ≈ U @ (V + G @ W)ᵀ
    
    where:
        - U: user factors (m x k)
        - V: item factors (n x k)
        - G: one-hot genres (n x d), if provided
        - W: learnable projection from genres → latent space (d x k)
        - m: number of users
        - n: number of items
        - d: number of genres
        - k: number of latent factors
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
        self.genre_factors: np.ndarray | None = None
        self.genre_norm: np.ndarray | None = None

    def fit(self, R: np.ndarray, genres: np.ndarray | None = None) -> ALS:
        """
        Fit ALS model to rating matrix.

        Args:
            R: (m x n) ratings matrix with np.nan for missing entries.
            genres: (n x d) genre one-hot/multi-hot matrix, or None.

        Returns:
            Self (allows chaining, e.g. model.fit(R).predict()).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # m users × n items
        m, n = R.shape
        
        # Normalize genres to have unit sum per item
        if genres is not None:
            self.genre_norm = np.maximum(genres.sum(axis=1, keepdims=True), 1)
            genres = genres / np.maximum(genres.sum(axis=1, keepdims=True), 1)

        # Random initialization of user/item factors with small Gaussian noise
        self.user_factors = np.random.normal(
            scale=SCALE_FACTOR,
            size=(m, self.n_factors)
            )
        self.item_factors = np.random.normal(
            scale=SCALE_FACTOR,
            size=(n, self.n_factors)
            )

        # If genre data is provided, initialize genre factors
        if genres is not None:
            d = genres.shape[1]
            self.genre_factors = np.random.normal(
                scale=SCALE_FACTOR,
                size=(d, self.n_factors)
                )
        else:
            self.genre_factors = None

        # Mask = which entries in R are observed
        # (True = observed, False = missing)
        mask = ~np.isnan(R)

        logger.info(
            f"Starting ALS training: m={m} users, n={n} items, "
            f"factors={self.n_factors}, n_iters={self.n_iters}, "
            f"reg={self.reg:.4f}, random_state={self.random_state}. "
            f"Genres {'included.' if genres is not None else 'not included.'}"
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

                # Factors of rated items
                if self.genre_factors is None:
                    V = self.item_factors[idx]
                else:
                    V = self.item_factors[idx] + genres[idx] @ self.genre_factors
                # Ratings given by user u
                r = R[u, idx]

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

                # If genre factors are used, subtract genre contribution
                if self.genre_factors is None:
                    A = U.T @ U + self.reg * np.eye(self.n_factors)
                    b = U.T @ r
                else:
                    genre_contrib = (self.user_factors[idx] @ self.genre_factors.T) @ genres[i]
                    r_tilde = r - genre_contrib
                    A = U.T @ U + self.reg * np.eye(self.n_factors)
                    b = U.T @ r_tilde

                # Solve normal equation: (UᵀU + λI) * V[i] = Uᵀr
                self.item_factors[i] = np.linalg.solve(A, b)

            # Update genre factors every 5 iterations (less noisy) if applicable
            if self.genre_factors is not None and (it % 5 == 0 or it == self.n_iters - 1):
                X_rows, y_vals = [], []
                for u in range(m):
                    for i in np.where(mask[u])[0]:
                        x = np.kron(genres[i], self.user_factors[u])
                        y = R[u, i] - self.user_factors[u] @ self.item_factors[i]
                        X_rows.append(x)
                        y_vals.append(y)
                X = np.vstack(X_rows)
                y = np.array(y_vals)

                # Use stronger regularization for genre factors
                A = X.T @ X + (10 * self.reg) * np.eye(X.shape[1])
                b = X.T @ y
                genre_vec = np.linalg.solve(A, b)
                self.genre_factors = genre_vec.reshape(genres.shape[1], self.n_factors)

            logger.debug("Completed iteration %d", it + 1)

        logger.info("ALS training finished.")

        return self
    
    def fit_laplacian(self, R: np.ndarray, genres=None, S=None, alpha=0.1) -> ALS:
        """
        Fits the ALS model with Laplacian regularization on item factors.

        This variant of ALS: a Laplacian penalty is added to encourage items that are similar in a given
        graph (here, items with similar genres) to have similar latent embeddings.

        Args:
            R (np.ndarray): (m × n) ratings matrix with NaN for missing entries.
            genres (np.ndarray | None): (n × d) optional one-hot or multi-hot item genre matrix.
            S (np.ndarray | None): (n × n) symmetric item similarity matrix used
                to compute the Laplacian (L = D - S).
            alpha (float): Strength of Laplacian regularization.

        Returns:
            self (ALS): Fitted ALS model with Laplacian regularization applied.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        m, n = R.shape
        mask = ~np.isnan(R)

        if self.user_factors is None:
            self.user_factors = np.random.normal(scale=SCALE_FACTOR, size=(m, self.n_factors))
        if self.item_factors is None:
            self.item_factors = np.random.normal(scale=SCALE_FACTOR, size=(n, self.n_factors))

        if genres is not None:
            d = genres.shape[1]
            self.genre_factors = np.random.normal(scale=SCALE_FACTOR, size=(d, self.n_factors))
            self.genre_norm = np.maximum(genres.sum(axis=1, keepdims=True), 1)
            genres = genres / self.genre_norm
        else:
            self.genre_factors = None

        if S is not None:
            D = S.sum(axis=1)
        else:
            D = np.zeros(n)

        logger.info(f"Starting Laplacian ALS training α={alpha}")

        for it in trange(self.n_iters, desc="Laplacian ALS"):
            # User update
            for u in range(m):
                idx = mask[u]
                if not np.any(idx):
                    continue
                if self.genre_factors is None:
                    V = self.item_factors[idx]
                else:
                    V = self.item_factors[idx] + genres[idx] @ self.genre_factors
                r = R[u, idx]
                A = V.T @ V + self.reg * np.eye(self.n_factors)
                b = V.T @ r
                self.user_factors[u] = np.linalg.solve(A, b)

            # Item update with Laplacian
            for i in range(n):
                idx = mask[:, i]
                if not np.any(idx):
                    continue
                U = self.user_factors[idx]
                r = R[idx, i]
                A = U.T @ U + (self.reg + alpha * D[i]) * np.eye(self.n_factors)
                b = U.T @ r
                if S is not None:
                    b += alpha * np.sum(S[i, :, None] * self.item_factors, axis=0)
                self.item_factors[i] = np.linalg.solve(A, b)

        logger.info("Laplacian ALS training finished.")
        return self

    def predict(self, genres) -> np.ndarray:
        """
        Predict the full rating matrix R̂ = U @ Vᵀ.

        Returns:
            Completed ratings matrix (np.ndarray).
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before prediction.")

        if self.genre_factors is None or genres is None:
            enriched_items = self.item_factors
        else:
            genres = genres / self.genre_norm
            enriched_items = self.item_factors + genres @ self.genre_factors

        return self.user_factors @ enriched_items.T


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
    test_path: str | None,
    genres_path: str | None,
    params: Dict[str, int | float],
    merge: bool,
    use_laplacian: bool = False,
) -> np.ndarray:
    """
    High-level helper function to complete ratings.

    Loads data, merges train/test, trains ALS, and returns predictions.

    Args:
        train_path: Path to training `.npy` file.
        test_path: Path to test `.npy` file.
        genres_path: Path to genres `.npy` file, or None.
        params: Model hyperparameters (dict with keys):
            - n_factors
            - n_iters
            - reg
            - random_state
            - (optional, for Laplacian regularization)
                - S_topk
                - S_eps
                - alpha
        merge: Whether to merge train and test (to train on all data before
               submission).
        use_laplacian: Whether to apply Laplacian regularization.

    Returns:
        Completed ratings matrix (np.ndarray).
    """
    # Load train and test sets
    R_train = read_data(train_path)

    # Merge them into a single observed matrix
    if merge:
        R_test = read_data(test_path)
        R_merged = merge_train_test(R_train, R_test)
    else:
        R_merged = R_train
    
    # Load genres if provided
    if genres_path is not None:
        genres = read_data(genres_path)
    else:
        genres = None

    # Initialize and fit ALS model
    model = ALS(
        n_factors=int(params.get("n_factors")),
        n_iters=int(params.get("n_iters")),
        reg=float(params.get("reg")),
        random_state=params.get("random_state"),
    )

    # Fit with or without Laplacian regularization
    if use_laplacian:
        S_topk = int(params.get("S_topk", 10))
        S_eps = float(params.get("S_eps", 1e-5))
        alpha = float(params.get("alpha", 0.1))
        if genres is not None:
            S = build_item_similarity_from_genres(genres, topk=S_topk, eps=S_eps)
        else:
            S = None
        model.fit_laplacian(R_merged, genres=None, S=S, alpha=alpha)
    else:
        model.fit(R_merged, genres=genres)

    return model.predict(genres=genres)
