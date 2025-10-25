"""
Alternating Least Squares (ALS) for collaborative filtering via matrix
factorization with optional feature side-information and graph (Laplacian)
regularization.

Overview
--------
We factorize a partially observed user-item rating matrix R (shape m x n):

    R ≈ U @ (V + sum_f(X_f @ W_f))^T + μ + b_u[:, None] + b_i[None, :]

where:
- U ∈ R^{m x k} are user latent factors,
- V ∈ R^{n x k} are base item factors,
- (V + sum_f(X_f @ W_f)) ∈ R^{n x k} are enriched item factors,
- X_f ∈ R^{n x d_f} are optional item feature matrices,
- W_f ∈ R^{d_f x k} are learned linear projections for each feature f,
- μ is the global mean,
- b_u ∈ R^m is user biases,
- b_i ∈ R^n is item biases.

We fit parameters by alternating ridge-regularized least squares over observed
entries only. Optionally, we regularize V with a graph Laplacian built from a
chosen feature (cosine similarity, top-k).

Key ideas
---------
- **Generic features**: pass any number of item feature matrices via a dict
  `features = {"genres": G, "emb": E, ...}`; each gets its own W_f and λ_w[f].
- **Popularity-aware regularization (items)**: λ_v can scale inversely with
  item popularity to avoid over-penalizing rare items.
- **Graph regularization (optional)**: construct item-item similarity S from
  a chosen feature (e.g., "genres"), keep top-k per row, and apply Laplacian
  regularization on V.

Quick start
-----------
>>> import numpy as np
>>> from scripts.als_config import (
...    ALSConfig, CoreConfig, BiasesConfig, GraphConfig, GraphSimConfig
...    )
>>> from scripts.als import ALS
>>>
>>> R = np.load("data/ratings_train.npy")   # (m, n), NaN = missing
>>> G = np.load("data/genres.npy")          # (n, d_g)
>>> Y = np.load("data/years.npy")           # (n, d_y)
>>>
>>> cfg = ALSConfig(
...     core=CoreConfig(
...         n_factors=32, n_iters=20, lambda_u=5.0, lambda_v=6.0,
...         pop_reg_mode="inverse_sqrt", random_state=42, update_w_every=5
...     ),
...     biases=BiasesConfig(
...         lambda_bu=3.0,
...         lambda_bi=2.0
...     ),
...     graph=GraphConfig(
...         alpha=0.5,  # enable Laplacian
...         sim=GraphSimConfig(
...             source="feature", feature_name="genres", metric="cosine",
...             topk=50, eps=1e-8
...         )
...     )
... )
>>>
>>> features = {"genres": G, "years": Y}        # any number of item features
>>> lambda_w = {"genres": 5.0, "years": 10.0}   # reg per feature
>>> model = ALS(lambda_w=lambda_w, config=cfg)
>>>
>>> model.fit(R, features=features)
>>> R_hat = model.predict(features=features)    # full m x n completed matrix

Notes
-----
- All feature matrices must have n rows (one row per item).
- Pass only preprocessed features (e.g., normalized/binned) — this module is
  not responsible for feature engineering.
- Graph regularization is applied only if `graph.alpha > 0` and the
  configured similarity feature exists in `features`.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from tqdm import trange

from scripts.als_config import ALSConfig
from scripts.helpers import cholesky_solve

# Numerical constants
SCALE_FACTOR = 0.1          # Scale for random init of factors
EPS = 1e-10                 # Small constant to avoid singularities

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


class ALS:
    """
    Alternating Least Squares (ALS) for matrix factorization with features.

    Factorizes a user-item rating matrix R into low-rank matrices:

        R ≈ U @ (V + sum_f(X_f @ W_f))^T + μ + b_u[:, None] + b_i[None, :]
    where:
        - U: User factors (m x k)
        - V: Item base factors (n x k)
        - X_f: Item feature matrices (n x d_f)
        - W_f: Feature projection matrices (d_f x k)
        - μ: Global mean (scalar)
        - b_u: User biases (m,)
        - b_i: Item biases (n,)

    Training (ALS loop):
        fit(R, features) learns U, V, W_f, μ, b_u, b_i by alternating updates:
            (1) Update U and b_u fixing others
            (2) Update V and b_i fixing others
            (3) Update each W_f fixing others
            (4) Update global mean μ

    Inference:
        predict(features) returns full predicted rating matrix R̂.

        R̂ = U @ (V + sum_f(X_f @ W_f))^T + μ + b_u[:, None] + b_i[None, :]

    """

    def __init__(
        self,
        config: ALSConfig,
        lambda_w: Dict[str, float] | None = None
    ) -> None:
        """
        Initialize ALS model.
        Args:
            config: ALSConfig dataclass with hyperparameters.
            lambda_w: Dict mapping feature names to regularization values.
        """
        # Configuration
        if config is None:
            raise ValueError("ALSConfig must be provided.")
        else:
            self.cfg = config

        # Learned projection matrices per feature (name -> W_f)
        self.W: Dict[str, np.ndarray] = {}
        self.lambda_w: Dict[str, float] = dict(lambda_w or {})

        # Core hyperparameters
        core = self.cfg.core
        self.n_factors = core.n_factors
        self.n_iters = core.n_iters
        self.lambda_u = core.lambda_u
        self.lambda_v = core.lambda_v
        self.random_state = core.random_state
        self.update_w_every = core.update_w_every
        self.pop_reg_mode = core.pop_reg_mode

        # Bias regularization (defaults fall back to λ_u/λ_v)
        self.lambda_bu = self.cfg.biases.lambda_bu or self.lambda_u
        self.lambda_bi = self.cfg.biases.lambda_bi or self.lambda_v

        # Graph config
        self.alpha = self.cfg.graph.alpha  # Laplacian strength ∈ [0, +∞)
        self.S_topk = (self.cfg.graph.sim.topk
                       if self.cfg.graph.sim is not None else None)
        self.S_eps = (self.cfg.graph.sim.eps
                      if self.cfg.graph.sim is not None else EPS)

        # Learned parameters (initialized in fit)
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.b_u: np.ndarray | None = None
        self.b_i: np.ndarray | None = None
        self.mu: float = 0.0
        self.S: np.ndarray | None = None

        # Training history
        self.history: Dict[str, list[float]] = {
            "train_rmse": [],
            "U_norm": [],
            "V_norm": [],
            "bu_norm": [],
            "bi_norm": [],
        }


    def _build_item_similarity_from_feature(
        self,
        features: Dict[str, np.ndarray],
        ) -> np.ndarray | None:
        """
        Build an item-item cosine similarity matrix S using the feature.
        
        Args:
            features: Dictionary of feature matrices (name -> X).

        Returns:
            S : Symmetric similarity (n x n) with zeros on the diagonal. Only
                the top-k neighbors per row are kept (if topk is set). Returns
                None if a similarity feature isn't configured or isn't present
                in `features`.
        """

        # Don't build S if no graph sim config
        if not self.cfg.graph or not self.cfg.graph.sim:
            return None

        f_name = self.cfg.graph.sim.feature_name
        X = features.get(f_name)

        # Check if feature matrix is present
        if X is None:
            logger.warning(f"GraphSim feature '{f_name}' not found in "
                           f"features dict. Graph regularization disabled.")
            return None

        # Cosine on rows of X
        norms = np.sqrt((X * X).sum(axis=1, keepdims=True)) + self.S_eps
        Xn = X / norms
        S = Xn @ Xn.T
        np.fill_diagonal(S, 0.0)

        # Keep only top-k per row
        top_k = self.S_topk
        if top_k is not None and top_k < S.shape[0]:
            n = S.shape[0]
            for i in range(n):
                idx = np.argpartition(S[i], -top_k)[:-top_k]
                S[i, idx] = 0.0

        # Symmetrize
        S = np.maximum(S, S.T)
        return S


    def _calculate_item_reg(self, counts: np.ndarray) -> np.ndarray:
        """Calculate item-specific regularization values based on popularity.

        Args:
            counts: Number of observed ratings for each item, shape (n,).

        Returns:
            Array of regularization values for each item.
        """
        # Return uniform regularization if no popularity mode is set
        if not self.pop_reg_mode:
            return np.full_like(counts, self.lambda_v, dtype=float)
        # Return popularity-based regularization
        elif self.pop_reg_mode == "inverse_sqrt":
            return self.lambda_v / np.sqrt(counts + 1.0)
        else:
            raise ValueError(f"Unknown pop_reg_mode '{self.pop_reg_mode}'")


    def _enrich_item_factors_with_features(
        self,
        idx_items: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute enriched item factors Z_i for selected items: 

            Z_i = V_i + Σ_f X_f[i] W_f.
        
        Args:
            idx_items: Indices of items to enrich.
            features: Dictionary of feature matrices (name -> X).

        Returns:
            Enriched item factors (len(idx_items) x k).
        """
        Z = self.V[idx_items].copy()
        for name, X in features.items():
            W = self.W.get(name)
            if W is not None:
                Z += X[idx_items] @ W
        return Z


    def fit(self,
            R: np.ndarray,
            features: Dict[str, np.ndarray] | None = None,
            ) -> ALS:
        """
        Fit the model to a ratings matrix with optional item features.

        Args:
            R: (m x n) ratings matrix with np.nan for missing entries.
            features: Dictionary of feature matrices (name -> X).

        Returns:
            The fitted model.

        Raises:
            ValueError: If feature matrices have incompatible shapes or
                        contain infinite values (e.g. NaN).
        """
        # Set random generator
        rng = np.random.default_rng(self.random_state)

        # m users × n items
        m, n = R.shape
        
        # Mask = which entries in R are observed
        mask = ~np.isnan(R)

        # Prepare feature matrices
        features = features or {}

        # Validate feature matrices
        for name, X in features.items():
            if X.shape[0] != n:
                raise ValueError(f"Feature '{name}' has {X.shape[0]} rows; "
                                 f"expected {n} (number of items).")
            if not np.isfinite(X).all():
                raise ValueError(f"Feature '{name}' contains infinite values.")

        # Graph regularization (if enabled and feature available)
        use_graph = (self.alpha > 0.0) and (self.S_topk is not None)
        self.S = self._build_item_similarity_from_feature(features) if use_graph else None
        use_graph = bool(self.S is not None)
        D = self.S.sum(axis=1) if use_graph else None

        # Initialize biases and factors
        self.mu = float(np.nanmean(R))
        self.b_u = np.zeros(m, dtype=float)
        self.b_i = np.zeros(n, dtype=float)
        k = self.n_factors
        I = np.eye(k)

        # Initialize factors with small Gaussian noise
        self.U = rng.normal(scale=SCALE_FACTOR, size=(m, k))
        self.V = rng.normal(scale=SCALE_FACTOR, size=(n, k))

        # Initialize projection matrices for each present feature
        for name, X in features.items():
            d = X.shape[1]
            self.W[name] = rng.normal(
                scale=SCALE_FACTOR,
                size=(d, k)
                )

        # Calculate item popularities for regularization
        item_counts = mask.sum(axis=0).astype(float)

        # Returns uniform or popularity-based regularization based on self.pop_reg_mode
        lambda_v_i = self._calculate_item_reg(item_counts).astype(float)
        # TODO: allow popularity-based reg for biases too
        lambda_bi_i = np.full(n, float(self.lambda_bi), dtype=float)    

        # Precompute observed coordinate pairs once (mask is static)
        rows_u, rows_i = np.where(mask)

        # Main ALS loop
        for it in trange(self.n_iters, desc="ALS iterations"):

            # (1) Update user factors
            for u in range(m):
                # Items that user u has rated
                idx = mask[u]
                # Skip users with no ratings
                if not np.any(idx):
                    continue

                # Enrich item factors for rated items (nu, k)
                Z = self._enrich_item_factors_with_features(
                    np.where(idx)[0],
                    features
                    )

                # Compute residuals for user u after removing biases and mean
                r = R[u, idx] - (self.mu + self.b_u[u] + self.b_i[idx])
                A = Z.T @ Z + (self.lambda_u + EPS) * I
                b = Z.T @ r
                self.U[u] = cholesky_solve(A, b)

                # Update user bias (ridge update, closed-form)
                denom = idx.sum() + self.lambda_bu + EPS
                pred_wo_bu = (Z @ self.U[u]) + self.mu + self.b_i[idx]
                self.b_u[u] = float(np.sum(R[u, idx] - pred_wo_bu) / denom)

            # (2) Update item factors
            for i in range(n):
                # Users that rated item i
                idx = mask[:, i]
                # Skip items with no ratings
                if not np.any(idx):
                    continue

                # Take user factors for users who rated item i
                U_i = self.U[idx]  # (nu, k)

                # Compute residuals for item i after removing biases and mean
                r = R[idx, i] - (self.mu + self.b_u[idx] + self.b_i[i])
                
                # Regularization term for item i
                reg_i = lambda_v_i[i] + EPS
                
                # Add Laplacian term if activated
                if use_graph:
                    reg_i += float(self.alpha) * float(D[i])

                A = U_i.T @ U_i + reg_i * I
                b = U_i.T @ r
                if use_graph:
                    b += float(self.alpha) * (self.S[i] @ self.V)
                
                # Update item factor by solving the linear system
                self.V[i] = cholesky_solve(A, b)

                # Update item bias (ridge update, closed-form)
                denom = idx.sum() + lambda_bi_i[i] + EPS
                pred_wo_bi = (U_i @ self.V[i]) + self.mu + self.b_u[idx]
                self.b_i[i] = float(np.sum(R[idx, i] - pred_wo_bi) / denom)

            if (features) and ((it % self.update_w_every == 0) or (it == self.n_iters - 1)):
                # Remove genre contribution from residuals: data - (mu + bu + bi + U@V^T)
                r_obs = R[rows_u, rows_i] - (self.mu + self.b_u[rows_u] + self.b_i[rows_i])
                r_obs = r_obs - np.sum(self.U[rows_u] * self.V[rows_i], axis=1)

                # Subtract previously-updated features so each feature sees its own residual
                residual = r_obs.copy()
                for name, X in features.items():
                    W = self.W.get(name)
                    if W is not None:
                        WX = X[rows_i] @ W
                        residual -= np.sum(self.U[rows_u] * WX, axis=1)

                # Gauss–Seidel: for each feature add back its own contribution and refit
                for name, X in features.items():
                    d = X.shape[1]
                    if name in self.W:
                        residual_plus_self = residual.copy()
                        WX = X[rows_i] @ self.W[name]
                        residual_plus_self += np.sum(self.U[rows_u] * WX, axis=1)
                    else:
                        residual_plus_self = residual

                    # Prepare design matrix for ridge regression
                    U_obs = self.U[rows_u]                      # (N_obs, k)
                    X_obs = X[rows_i]                           # (N_obs, d)
                    # Design matrix for ridge: vec(U ⊗ X), shape (N_obs, d_f * k)
                    X_design = (X_obs[:, :, None] * U_obs[:, None, :]).reshape(len(rows_u), d * k)

                    lam = float(self.lambda_w.get(name, 0.0))
                    A = X_design.T @ X_design + (lam + EPS) * np.eye(d * k)
                    b = X_design.T @ residual_plus_self
                    vec = cholesky_solve(A, b)
                    self.W[name] = vec.reshape(d, k)

            # (4) Update global mean μ
            Z_rows = self._enrich_item_factors_with_features(
                rows_i,
                features
                )
            pred_wo_mu = (np.sum(self.U[rows_u] * Z_rows, axis=1) 
                          + self.b_u[rows_u] + self.b_i[rows_i])
            self.mu = float(np.mean(R[rows_u, rows_i] - pred_wo_mu))
            
            # Save training history
            pred = pred_wo_mu + self.mu
            err = R[rows_u, rows_i] - pred
            rmse = float(np.sqrt(np.mean(err ** 2)))
            self.history["train_rmse"].append(rmse)
            self.history["U_norm"].append(float(np.linalg.norm(self.U)))
            self.history["V_norm"].append(float(np.linalg.norm(self.V)))
            self.history["bu_norm"].append(float(np.linalg.norm(self.b_u)))
            self.history["bi_norm"].append(float(np.linalg.norm(self.b_i)))

        logger.info(f"ALS training finished. "
                    f"Final train RMSE: {self.history['train_rmse'][-1]:.4f}")

        return self


    def predict(
        self,
        features: Dict[str, np.ndarray] | None = None
        ) -> np.ndarray:
        """
        Compute the completed rating matrix R̂.
            R̂ = U @ (V + Σ_f X_f W_f)^T + μ + b_u[:, None] + b_i[None, :]
        
        Args:
            features: Dictionary of feature matrices (name -> X).

        Returns:
            Completed rating matrix of shape (m, n).

        Raises:
            RuntimeError: If model is not yet fitted.
            ValueError: If feature matrices have incompatible shapes.

        Example:
            R_hat = model.predict(features={"genres": G, "years": Y})
        """
        # Ensure model is fitted
        if self.U is None or self.V is None:
            raise RuntimeError("Model must be fitted before prediction.")

        # Validate feature matrices
        n = self.V.shape[0]
        features = features or {}
        for name, X in features.items():
            if X.shape[0] != n:
                raise ValueError(f"Feature '{name}' has {X.shape[0]} rows. "
                                 f"Expected number of rows: {n}.")
            if not np.isfinite(X).all():
                raise ValueError(f"Feature '{name}' contains infinite values.")

        # Compute enriched item factors Z = V + Σ_f X_f W_f
        Z = self.V.copy()
        for name, X in features.items():
            W = self.W.get(name)
            if W is not None:
                Z += X @ W

        return self.U @ Z.T + self.mu + self.b_u[:, None] + self.b_i[None, :]