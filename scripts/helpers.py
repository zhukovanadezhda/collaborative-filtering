from scipy.linalg import cho_factor, cho_solve
import numpy as np


def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def merge_train_test(
    R_train: np.ndarray,
    R_test: np.ndarray
    ) -> np.ndarray:
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