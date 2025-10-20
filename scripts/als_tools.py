"""
Alternating Least Squares (ALS) for collaborative filtering.

Implements matrix factorization with regularized least squares to impute
missing ratings. 
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Any
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from tqdm import trange

from scripts.laplacian_tools import build_item_similarity_from_genres

# Scale factor for random initialization of latent factors
SCALE_FACTOR = 0.1
# Small epsilon for numerical stability (to ensure SPD matrices)
EPS_SPD = 1e-10

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def _cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b with Cholesky.

    Args:
        A: Symmetric positive definite matrix.
        b: Right-hand side vector.

    Returns:
        Solution vector x.

    Note:
        In classic LS we'd prefer QR/SVD to avoid forming X^T@X explicitly, but
        if A is small and SPD, Cholesky is preferred.
    """
    L, lower = cho_factor(A, check_finite=False)
    return cho_solve((L, lower), b, check_finite=False)


def _build_year_design(
    years: np.ndarray | None,
    mode: str | None = None,
    n_bins: int = 10,
) -> Tuple[np.ndarray | None, Dict[str, float]]:
    """
    Build per-item year design matrix Y:
    - 'cont'  -> (n x 1) centered & standardized column
    - 'bins'  -> (n x n_bins) one-hot with quantile binning
    
    Args:
        years: (n,) array of item years, or None.
        mode: 'cont', 'bins', or None.
        n_bins: Number of bins if mode='bins'.

    Returns:
        Y: Year design matrix, or None.
        stats: Dict of statistics used for transformation.
    """
    if years is None:
        return None, {}

    years = years.astype(float)
    stats: Dict[str, float] = {}

    if mode == "cont":
        mu = np.nanmean(years)
        sd = max(np.nanstd(years), 1e-8)
        y = (years - mu) / sd
        # 
        y[~np.isfinite(y)] = 0.0
        stats.update({"mu": mu, "sd": sd})
        return y[:, None], stats

    if mode == "bins":
        # Quantile bins, robust to skew
        quantiles = np.quantile(years, np.linspace(0, 1, n_bins + 1))
        # Ensure strictly increasing edges
        for i in range(1, len(quantiles)):
            if quantiles[i] <= quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1] + EPS_SPD
        # Build one-hot encoding
        idx = np.full(years.shape[0], 0, dtype=int)
        finite = np.isfinite(years)
        idx[finite] = np.clip(np.digitize(years[finite], quantiles[1:-1], right=True), 0, n_bins - 1)
        Y = np.zeros((years.shape[0], n_bins), dtype=float)
        Y[np.arange(years.shape[0]), idx] = 1.0
        stats.update({"edges": quantiles})
        return Y, stats

    raise ValueError(f"Unknown year_mode '{mode}'")

class ALS:
    """
    Alternating Least Squares (ALS) for collaborative filtering.

    Factorizes a user-item rating matrix R into three low-rank matrices:

    R ≈ U @ (V + G @ W_g + Y @ W_y)^T + 
        + μ + b_u.reshape(-1, 1) + b_i.reshape(1, -1)

    by minimizing squared error on observed entries with ridge regularizations:
    
    min_{U,V,W_g,W_y,b_u,b_i,μ}  Σ_{(u,i) ∈ Ω} 
        ( R_{u,i} - [ U_u @ ( V_i + G_i @ W_g + Y_i @ W_y ) + μ + b_u[u] + b_i[i] ] )^2
        + λ_u ||U|| + λ_v ||V|| + λ_wg ||W_g|| + λ_wy ||W_y||

    where:
        - U    ∈ R^{m x k}  : user latent factors
        - V    ∈ R^{n x k}  : item latent factors
        - G    ∈ R^{n x d}  : item genres (one/multi-hot, row-normalized), optional
        - W_g  ∈ R^{d x k}  : learnable projection from genres → latent space
        - Y    ∈ R^{n x p}  : year features (continuous or binned), optional
                - year_mode = "cont"  -> p = 1 (standardized continuous year)
                - year_mode = "bins"  -> p = n_year_bins (one-hot quantile bins)
        - W_y  ∈ R^{p x k}  : learnable projection from year → latent space
        - μ    ∈ R          : global mean
        - b_u  ∈ R^{m}      : user biases
        - b_i  ∈ R^{n}      : item biases
        - m                 : number of users
        - n                 : number of items
        - d                 : number of genres
        - p                 : number of year features
        - k                 : number of latent factors
        - λ_u  ∈ R          : regularization on U
        - λ_v  ∈ R          : regularization on V (optionally dependent on item popularity)
        - λ_wg ∈ R          : regularization on W_g
        - λ_wy ∈ R          : regularization on W_y

    Training (ALS loop):
        1) Fix V, W_g, W_y, b_i → update U and b_u
        2) Fix U, W_g, W_y, b_u → update V and b_i
        3) Periodically update W_g and W_y by ridge on observed entries
        4) Update μ as the mean residual over observed entries

    Inference:
        predict(genres, years) returns:
            R_hat = U @ (V + G W_g + Y W_y).T + 
                    + μ + b_u.reshape(1, -1) + b_i.reshape(1, -1)
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_iters: int = 50,
        lambda_u: float = 0.8,
        lambda_v: float = 0.8,
        lambda_wg: float = 100.0,
        lambda_wy: float = 1.0,
        random_state: int | None = 42,
        year_mode: str | None = None,
        n_year_bins: int | None = None,
        pop_reg_mode: str | None = None,
        update_w_every: int = 5
    ) -> None:
        """
        Initialize ALS model.

        Args:
            n_factors: Number of latent factors (k).
            n_iters: Number of ALS iterations.
            lambda_u: Regularization for user factors U.
            lambda_v: Regularization for item factors V.
            lambda_wg: Regularization for genre projection W_g.
            lambda_wy: Regularization for year projection W_y.
            random_state: Random seed for reproducibility.
            year_mode: How to handle year features ('cont', 'bins').
            n_year_bins: Number of bins if year_mode='bins'.
            pop_reg_mode: Popularity-based reg for items (c).
            update_w_every: Frequency of updating W_g and W_y (in iterations).
        """
        
        # Hyperparameters
        self.n_factors = int(n_factors)
        self.n_iters = int(n_iters)
        self.pop_reg_mode = pop_reg_mode
        self.lambda_u = float(lambda_u)
        self.lambda_v = float(lambda_v)
        self.lambda_wg = float(lambda_wg)
        self.lambda_wy = float(lambda_wy)
        self.random_state = random_state
        self.year_mode = year_mode
        self.n_year_bins = n_year_bins
        self.update_w_every = int(update_w_every)
        
        # Learned parameters (to be filled during training)        
        self.U: np.ndarray | None = None           # (m, k)
        self.V: np.ndarray | None = None           # (n, k)
        self.Wg: np.ndarray | None = None          # (d, k) genres → latent
        self.Wy: np.ndarray | None = None          # (p, k) years  → latent
        self.mu: float = 0.0
        self.b_u: np.ndarray | None = None         # (m,)
        self.b_i: np.ndarray | None = None         # (n,)

        # Precomputed norms / transforms
        self.genre_norm: np.ndarray | None = None  # (n, 1)
        self.year_stats: Dict[str, float] = {}     # for standardization if year_mode='cont'

    
    def _calculate_item_reg(self, counts: np.ndarray) -> np.ndarray:
        """Calculate item-specific regularization values based on popularity.

        Args:
            counts: Array of item popularity counts.

        Returns:
            Array of regularization values for each item.
        """
        if not self.pop_reg_mode:
            return np.full_like(counts, self.lambda_v, dtype=float)
        elif self.pop_reg_mode == "inverse_sqrt":
            return self.lambda_v / np.sqrt(counts + 1.0)
        else:
            raise ValueError(f"Unknown pop_reg_mode '{self.pop_reg_mode}'")

    def _enrich_item_factors(self,
                             idx_items: np.ndarray,
                             G: np.ndarray | None,
                             Y: np.ndarray | None) -> np.ndarray:
        """Enrich item factors V_i with genre and year contributions.

        Args:
            idx_items: Indices of items to enrich.
            G: Genre matrix (n x d) or None.
            Y: Year feature matrix (n x p) or None.

        Returns:
            Enriched item factors (len(idx_items) x k).
        """
        Z = self.V[idx_items].copy()
        if (G is not None) and (self.Wg is not None):
            Z += G[idx_items] @ self.Wg
        if (Y is not None) and (self.Wy is not None):
            Z += Y[idx_items] @ self.Wy
        return Z


    def fit(self,
            R: np.ndarray,
            genres: np.ndarray | None = None,
            years: np.ndarray | None = None
            ) -> ALS:
        """
        Fit ALS model to rating matrix.

        Args:
            R: (m x n) ratings matrix with np.nan for missing entries.
            genres: (n x d) genre one-hot/multi-hot matrix, or None.
            years: (n,) array of item years, or None.

        Returns:
            Self (allows chaining, e.g. model.fit(R).predict()).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # m users × n items
        m, n = R.shape
        
        # Mask = which entries in R are observed
        mask = ~np.isnan(R)

        # Initialize biases to zero
        self.mu = float(np.nanmean(R))
        self.b_u = np.zeros(m, dtype=float)
        self.b_i = np.zeros(n, dtype=float)
        
        # Normalize genres to have unit sum per item
        G = None
        if genres is not None:
            self.genre_norm = np.maximum(genres.sum(axis=1, keepdims=True), 1.0)
            G = genres / self.genre_norm

        # Build year design Y
        Y = None
        if years is not None:
            Y, self.year_stats = _build_year_design(years,
                                                    self.year_mode,
                                                    self.n_year_bins)

        # Initialize factors with small Gaussian noise
        k = self.n_factors

        self.U = np.random.normal(
            scale=SCALE_FACTOR,
            size=(m, k)
            )
        self.V = np.random.normal(
            scale=SCALE_FACTOR,
            size=(n, k)
            )

        # If genre data is provided, initialize genre factor
        if G is not None:
            d = genres.shape[1]
            self.Wg = np.random.normal(
                scale=SCALE_FACTOR,
                size=(d, k)
                )
        else:
            self.Wg = None

        # If years are provided, initialize years factor
        if Y is not None:
            p = Y.shape[1]
            self.Wy = np.random.normal(
                scale=SCALE_FACTOR,
                size=(p, k)
                )
        else:
            self.Wy = None

        # Calculate item popularities for regularization
        item_counts = mask.sum(axis=0).astype(float)
        if self.pop_reg_mode is None:
            # Uniform regularization
            lambda_v_i = np.full(n, self.lambda_v, dtype=float)
        else:
            # Popularity-based regularization
            lambda_v_i = self._calculate_item_reg(item_counts)

        # Precompute regularization matrix for factors
        I = np.eye(k) 

        # Set up logger info
        logger.info(
            f"Starting ALS training: m={m} users, n={n} items, "
            f"n_factors={self.n_factors}, "
            f"n_iters={self.n_iters}, reg_u={self.lambda_u}, "
            f"reg_v={self.lambda_v}, reg_wg={self.lambda_wg}, "
            f"reg_wy={self.lambda_wy}. "
            f"Genres {'included.' if genres is not None else 'not included.'} "
            f"Years {'included.' if years is not None else 'not included.'} "
        )

        # Main ALS loop
        for it in trange(self.n_iters, desc="ALS iterations"):

            # (1) Update user factors
            for u in range(m):
                # Items that user u has rated
                idx = mask[u]
                # Skip users with no ratings
                if not np.any(idx):
                    continue

                # Enrich item factors for rated items
                Z = self._enrich_item_factors(np.where(idx)[0], G, Y) # (nu,k)

                # Compute residuals for user u after removing biases and mean
                r = R[u, idx] - (self.mu + self.b_u[u] + self.b_i[idx])
                A = Z.T @ Z + (self.lambda_u + EPS_SPD) * I
                b = Z.T @ r
                self.U[u] = _cholesky_solve(A, b)

                # Update user bias (one-step ridge on scalar)
                denom = idx.sum() + 1e-8 + self.lambda_u
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
                U_i = self.U[idx]  # (nu,k)

                # Compute residuals for item i after removing biases and mean
                r = R[idx, i] - (self.mu + self.b_u[idx] + self.b_i[i])
                A = U_i.T @ U_i + (lambda_v_i[i] + EPS_SPD) * I
                b = U_i.T @ r
                self.V[i] = _cholesky_solve(A, b)

                # Update item bias (one-step ridge on scalar)
                denom = idx.sum() + EPS_SPD + lambda_v_i[i]
                pred_wo_bi = (U_i @ self.V[i]) + self.mu + self.b_u[idx]
                self.b_i[i] = float(np.sum(R[idx, i] - pred_wo_bi) / denom)

            # (3) Update feature projections Wg, Wy periodically
            if (self.Wg is not None or self.Wy is not None) and (
                (it % self.update_w_every == 0) or (it == self.n_iters - 1)
            ):
                # Compute residuals on all observed entries
                rows_u, rows_i = np.where(mask)
                r_obs = R[rows_u, rows_i] - (self.mu + self.b_u[rows_u] + self.b_i[rows_i])
                # Remove current latent factor contribution
                r_obs = r_obs - np.sum(self.U[rows_u] * self.V[rows_i], axis=1)

                # Update Wg (genres)
                if self.Wg is not None and G is not None:
                    # Xg: for each obs, line = vec( U_u ⊗ G_i )  -> shape (N_obs, d*k)
                    U_obs = self.U[rows_u]             # (N_obs, k)
                    G_obs = G[rows_i]                  # (N_obs, d)
                    Xg = (G_obs[:, :, None] * U_obs[:, None, :]).reshape(len(rows_u), -1)  # (N_obs, d*k)
                    A = Xg.T @ Xg + (self.lambda_wg + EPS_SPD) * np.eye(Xg.shape[1])
                    b = Xg.T @ r_obs
                    vec = _cholesky_solve(A, b)
                    self.Wg = vec.reshape(G.shape[1], self.n_factors)

                # Update Wy (years)
                if self.Wy is not None and Y is not None:
                    # Remove genre contribution from residuals
                    r_tmp = r_obs.copy()
                    if self.Wg is not None and G is not None:
                        Gw = G[rows_i] @ self.Wg       # (N_obs, k)
                        r_tmp = r_tmp - np.sum(self.U[rows_u] * Gw, axis=1)

                    U_obs = self.U[rows_u]             # (N_obs, k)
                    Y_obs = Y[rows_i]                  # (N_obs, p)
                    Xy = (Y_obs[:, :, None] * U_obs[:, None, :]).reshape(len(rows_u), -1)  # (N_obs, p*k)
                    A = Xy.T @ Xy + (self.lambda_wy + EPS_SPD) * np.eye(Xy.shape[1])
                    b = Xy.T @ r_tmp
                    vec = _cholesky_solve(A, b)
                    self.Wy = vec.reshape(Y.shape[1], self.n_factors)

            logger.debug("Completed iteration %d", it + 1)

        logger.info("ALS training finished.")

        return self
    

    def fit_laplacian(
        self,
        R: np.ndarray,
        genres: np.ndarray | None = None,
        S: np.ndarray | None = None,
        alpha: float = 0.1,
    ) -> ALS:
        """
        Fits the ALS model with Laplacian regularization on item factors.

        This variant of ALS: a Laplacian penalty is added to encourage items that are similar in a given
        graph (here, items with similar genres) to have similar latent embeddings.

        Args:
            R: (m x n) ratings matrix with NaN for missing entries.
            genres: (n x d) optional one-hot or multi-hot item genre matrix.
            S: (n x n) symmetric item similarity matrix used to compute the
                       Laplacian (L = D - S).
            alpha: Strength of Laplacian regularization.

        Returns:
            self: Fitted ALS model with Laplacian regularization applied.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        m, n = R.shape
        mask = ~np.isnan(R)

        # Initialize biases around global mean
        self.mu = float(np.nanmean(R))
        self.b_u = np.zeros(m, dtype=float)
        self.b_i = np.zeros(n, dtype=float)

        # Normalize genres if provided
        G = None
        self.genre_norm = None
        if genres is not None:
            self.genre_norm = np.maximum(genres.sum(axis=1, keepdims=True), 1.0)
            G = genres / self.genre_norm

        # Reset learned parameters to align with current training call
        k = self.n_factors
        self.U = np.random.normal(scale=SCALE_FACTOR, size=(m, k))
        self.V = np.random.normal(scale=SCALE_FACTOR, size=(n, k))
        if G is not None:
            self.Wg = np.zeros((G.shape[1], k), dtype=float)
        else:
            self.Wg = None
        self.Wy = None
        self.year_stats = {}

        # Item-specific regularization (e.g., popularity based)
        item_counts = mask.sum(axis=0).astype(float)
        lambda_v_i = self._calculate_item_reg(item_counts)

        if S is not None:
            D = S.sum(axis=1)
        else:
            D = np.zeros(n, dtype=float)

        I = np.eye(k)

        logger.info(f"Starting Laplacian ALS training α={alpha}")

        for it in trange(self.n_iters, desc="Laplacian ALS"):
            # Update user factors and biases
            for u in range(m):
                idx_items = np.where(mask[u])[0]
                if idx_items.size == 0:
                    continue

                Z = self._enrich_item_factors(idx_items, G, None)
                r = R[u, idx_items] - (self.mu + self.b_u[u] + self.b_i[idx_items])
                A = Z.T @ Z + (self.lambda_u + EPS_SPD) * I
                b = Z.T @ r
                self.U[u] = _cholesky_solve(A, b)

                denom = idx_items.size + self.lambda_u + EPS_SPD
                pred_wo_bu = (Z @ self.U[u]) + self.mu + self.b_i[idx_items]
                self.b_u[u] = float(np.sum(R[u, idx_items] - pred_wo_bu) / denom)

            # Update item factors with Laplacian regularization and item biases
            for i in range(n):
                idx_users = np.where(mask[:, i])[0]
                if idx_users.size == 0:
                    continue

                U_i = self.U[idx_users]
                r = R[idx_users, i] - (self.mu + self.b_u[idx_users] + self.b_i[i])
                reg = lambda_v_i[i] + alpha * D[i] + EPS_SPD
                A = U_i.T @ U_i + reg * I
                b = U_i.T @ r
                if S is not None:
                    b += alpha * (S[i] @ self.V)
                self.V[i] = _cholesky_solve(A, b)

                denom = idx_users.size + lambda_v_i[i] + EPS_SPD
                pred_wo_bi = (U_i @ self.V[i]) + self.mu + self.b_u[idx_users]
                self.b_i[i] = float(np.sum(R[idx_users, i] - pred_wo_bi) / denom)

            # Update genre projection periodically
            if self.Wg is not None and G is not None and (
                (it % self.update_w_every == 0) or (it == self.n_iters - 1)
            ):
                rows_u, rows_i = np.where(mask)
                r_obs = R[rows_u, rows_i] - (self.mu + self.b_u[rows_u] + self.b_i[rows_i])
                r_obs = r_obs - np.sum(self.U[rows_u] * self.V[rows_i], axis=1)

                U_obs = self.U[rows_u]
                G_obs = G[rows_i]
                Xg = (G_obs[:, :, None] * U_obs[:, None, :]).reshape(len(rows_u), -1)
                A = Xg.T @ Xg + (self.lambda_wg + EPS_SPD) * np.eye(Xg.shape[1])
                b = Xg.T @ r_obs
                vec = _cholesky_solve(A, b)
                self.Wg = vec.reshape(G.shape[1], self.n_factors)

        logger.info("Laplacian ALS training finished.")
        return self

    def predict(self,
                genres: np.ndarray | None = None,
                years: np.ndarray | None = None
                ) -> np.ndarray:
            """
            Return full prediction matrix R̂ = U @ (V + G Wg + Y Wy)^T + biases.

            Returns:
                Completed ratings matrix (np.ndarray).
            """
            if self.U is None or self.V is None:
                raise RuntimeError("Model must be fitted before prediction.")

            n = self.V.shape[0]
            Z = self.V.copy()

            # Normalize genres and add contribution if provided
            if genres is not None and self.Wg is not None:
                if self.genre_norm is not None:
                    G = genres / np.maximum(self.genre_norm, 1.0)
                else:
                    G = genres / np.maximum(genres.sum(axis=1, keepdims=True), 1.0)
                # Add genre contribution
                Z += G @ self.Wg

            # Build year design and add contribution if provided
            if years is not None and self.Wy is not None:
                # Treat years as continuous variable
                if self.year_mode == "cont":
                    mu, sd = self.year_stats["mu"], self.year_stats["sd"]
                    y = (years.astype(float) - mu) / sd
                    y[~np.isfinite(y)] = 0.0
                    Y = y[:, None]
                # Treat years as binned categorical variable
                else:
                    edges = self.year_stats["edges"]
                    # Default bin 0 for NaNs
                    idx = np.full(years.shape[0], 0, dtype=int)
                    finite = np.isfinite(years)
                    idx[finite] = np.clip(
                        np.digitize(years[finite], edges[1:-1], right=True),
                        0,
                        len(edges)-2
                        )
                    Y = np.zeros((n, len(edges)-1), dtype=float)
                    Y[np.arange(n), idx] = 1.0
                # Add year contribution
                Z += Y @ self.Wy

            return self.U @ Z.T + self.mu + self.b_u[:, None] + self.b_i[None, :]


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
    years_path: str | None,
    params: dict[str, int | float | str | bool],
    merge: bool,
    use_laplacian: bool = False
) -> np.ndarray:
    """
    High-level helper function to complete ratings.

    Args:
        train_path : path to training `.npy`
        test_path  : path to test `.npy` (required if merge=True)
        genres_path: path to genres `.npy` (n_items x d_g), or None
        years_path  : path to years `.npy` (n_items,), or None
        params     : hyperparameters dict. Keys (all optional, sensible defaults):
            - n_factors: int
            - n_iters: int
            - random_state: int
            - lambda_u: float
            - lambda_v: float
            - lambda_wg: float
            - lambda_wy: float
            - year_mode: "cont" | "bins"
            - n_year_bins: int
            - pop_reg_mode: "inverse_sqrt"
            - update_w_every: int
            - (Laplacian) S_topk: int, S_eps: float, alpha: float
        merge       : if True, merge TRAIN+TEST observed entries before training
        use_laplacian: if True, use Laplacian ALS variant (uses genres graph)

    Returns:
        np.ndarray of shape (m, n) with completed ratings.
    """
    # Load data
    R_train = read_data(train_path)

    # Merge them into a single observed matrix
    if merge:
        if test_path is None:
            raise ValueError("merge=True requires `test_path`.")
        R_test = read_data(test_path)
        R_merged = merge_train_test(R_train, R_test)
    else:
        R_merged = R_train

    # Load genres and years if provided
    genres = read_data(genres_path) if genres_path is not None else None
    years  = read_data(years_path)  if years_path  is not None else None

    # Initialize and fit ALS model
    model = ALS(
        n_factors      = int(params.get("n_factors")),
        n_iters        = int(params.get("n_iters")),
        lambda_u       = float(params.get("lambda_u")),
        lambda_v       = float(params.get("lambda_v")),
        lambda_wg      = float(params.get("lambda_wg")),
        lambda_wy      = float(params.get("lambda_wy")),
        random_state   = int(params.get("random_state", 42)),
        year_mode      = str(params.get("year_mode", "cont")),
        n_year_bins    = int(params.get("n_year_bins", 10)),
        pop_reg_mode   = params.get("pop_reg_mode", None),
        update_w_every = int(params.get("update_w_every", 5)),
    )

    if use_laplacian:
        S_topk = int(params.get("S_topk"))
        S_eps  = float(params.get("S_eps"))
        alpha  = float(params.get("alpha"))
        if genres is not None:
            S = build_item_similarity_from_genres(genres, topk=S_topk, eps=S_eps)
        else:
            S = None
        model.fit_laplacian(R_merged, genres=None, S=S, alpha=alpha) # genres direct addition to MF deactivated 
    else:
        model.fit(R_merged, genres=genres, years=years)

    return model.predict(genres=genres, years=years)
