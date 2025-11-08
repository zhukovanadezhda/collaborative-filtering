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
