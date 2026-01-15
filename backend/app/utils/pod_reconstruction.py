import numpy as np


def reconstruct_pod_field(
    *,
    N: int,
    Psi: np.ndarray,
    A: np.ndarray,
    Xmean: np.ndarray,
    wdNorm: np.ndarray,
    idx: np.ndarray,
    ws_ref: float,
    wd_ref: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic POD reconstruction using the following formular:
    U = (Psi @ A + Xmean) * ws_ref
    """        

    # Interpolate coefficients (direction)
    AInterp = np.array([
        np.interp(wd_ref, wdNorm, A[i, :], period=360)
        for i in range(A.shape[0])
    ])

    # Build subset for selected ids (stacked data in Psi and Xmean)
    subset_idx_stacked = np.concatenate([idx, idx + N, idx + 2*N])

    # Reconstruct data for the calculated subset
    Psi_subset = Psi[subset_idx_stacked, :]
    Xmean_subset = Xmean[subset_idx_stacked]
    U_subset = (Psi_subset @ AInterp + Xmean_subset) * ws_ref

    # Split components for result
    Ux, Uy, Uz = np.split(U_subset, 3)
    return Ux.astype(np.float32), Uy.astype(np.float32), Uz.astype(np.float32)
