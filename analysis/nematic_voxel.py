from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Dict
import numpy as np

"""Nematic smoothing helper functions extracted from vis.py.

This module provides small utilities used for local nematic smoothing of
unit vectors: unit normalization, Gaussian neighbor weighting, building
the nematic (structure) tensor and extracting its principal axis.

The functions keep the same internal names used by the callers in
`vis.py` so the change is minimal (we import these names there).
"""
from typing import Tuple
import numpy as np
from numpy.linalg import eig


def _unit(v, eps: float = 1e-12) -> np.ndarray:
    """Row-wise normalize vector array `v`.

    v may be shape (M,3) or (3,), this returns the normalized vectors with
    the same trailing shape. Small norms are stabilized by `eps`.
    """
    v = np.asarray(v, float)
    # handle both (M,3) and (3,) shapes
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / max(n, eps)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _gaussian_weights(d2: np.ndarray, sigma: float) -> np.ndarray:
    """Return Gaussian weights for squared distances `d2` and std `sigma`.

    d2 can be a 1-D array of squared distances for a single center to many
    neighbors. The function returns exp(-0.5 * d2 / sigma**2).
    """
    sigma = float(sigma)
    return np.exp(-0.5 * d2 / (sigma ** 2))


def _nematic_weighted(vecs_unit: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute the weighted nematic (3x3) tensor from unit vectors.

    Parameters
    ----------
    vecs_unit : (M,3) array
        Unit direction vectors for the neighbor points.
    weights : (M,) array
        Non-negative weights for each neighbor.

    Returns
    -------
    N : (3,3) array
        Nematic tensor N = 1.5*(<a a^T>_w - I/3).
    """
    I = np.eye(3)
    aa = np.einsum("mi,mj->mij", vecs_unit, vecs_unit)  # (M,3,3)
    wsum = np.sum(weights) + 1e-12
    mean_aa = (weights[:, None, None] * aa).sum(axis=0) / wsum
    N = 1.5 * (mean_aa - I / 3.0)
    return N


def _principal_axis(N: np.ndarray) -> np.ndarray:
    """Return the principal eigenvector (unit) of a symmetric 3x3 tensor N."""
    vals, vecs = eig(N)
    v = vecs[:, int(np.argmax(vals.real))].real
    # ensure unit length
    return _unit(v.ravel())

def locally_average_nematic_vectors(
    V: np.ndarray,
    d2: np.ndarray,
    smoothing_std_um: float
) -> np.ndarray:
    """Smooth a field of unit vectors V using local nematic averaging."""
    V_s = np.zeros_like(V)
    for i in range(len(V)):
        w = _gaussian_weights(d2[i], smoothing_std_um)
        N = _nematic_weighted(V, w)
        V_s[i] = _principal_axis(N)
    return V_s

__all__ = ["_unit", "_gaussian_weights", "_nematic_weighted", "_principal_axis", "locally_average_nematic_vectors"]


def locally_average_nematic_vectors_from_dataframe(
    df,
    base: str,
    smoothing_std_um: float,
    sigma_column: str = None,
    com_units: str = "nm",
):
    """Convenience wrapper: compute locally-averaged nematic vectors from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing COM and base vector columns.
    base : str
        Base name used to find columns like '<base> X', '<base> Y', '<base> Z'.
    smoothing_std_um : float
        Smoothing standard deviation in µm. If None or <=0, returns normalized vectors.
    sigma_column : str, optional
        Optional per-row sigma column that must be finite to count as valid.
    com_units : str
        Units string for COM columns (default 'nm'). Returned COMs are converted to µm.

    Returns
    -------
    dict with keys:
      'valid_mask' : (N,) bool array of which rows were valid
      'V' : (N,3) array of smoothed unit vectors with NaNs for invalid rows
      'COM_um' : (N,3) array of COM coordinates in µm
    """
    import pandas as _pd

    if not isinstance(df, _pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    # We simplify assumptions: COM columns are exactly 'COM X (nm)', 'COM Y (nm)', 'COM Z (nm)'
    com_x, com_y, com_z = "COM X (nm)", "COM Y (nm)", "COM Z (nm)"
    if not all(c in df.columns for c in (com_x, com_y, com_z)):
        raise KeyError("Required COM columns not found: 'COM X (nm)', 'COM Y (nm)', 'COM Z (nm)'")
    # find base columns
    def _pick(col_suffix: str):
        target = f"{base} {col_suffix}"
        for c in df.columns:
            if c == target:
                return c
        target_low = target.lower()
        for c in df.columns:
            if c.lower() == target_low:
                return c
        raise KeyError(f"Missing column '{target}' in DataFrame")

    cx, cy, cz = _pick("X"), _pick("Y"), _pick("Z")

    # COM values are provided in nanometers; convert to micrometers for smoothing
    COM = df[[com_z, com_y, com_x]].to_numpy(float)
    COM_um = COM * 1e-3

    V_all = df[[cz, cy, cx]].to_numpy(float)
    valid = np.all(np.isfinite(V_all), axis=1)
    if sigma_column is not None:
        if sigma_column not in df.columns:
            raise KeyError(f"Specified sigma_column '{sigma_column}' not found in DataFrame.")
        valid &= np.isfinite(df[sigma_column].to_numpy(float))

    V_out = np.full_like(V_all, np.nan, dtype=float)

    if smoothing_std_um is None or smoothing_std_um <= 0:
        V_out[valid] = _unit(V_all[valid])
        return {"valid_mask": valid, "V": V_out, "COM_um": COM_um}

    Pv = COM_um[valid]
    if Pv.shape[0] == 0:
        return {"valid_mask": valid, "V": V_out, "COM_um": COM_um}

    d2_full = np.sum((Pv[:, None, :] - Pv[None, :, :]) ** 2, axis=-1)
    V = _unit(V_all[valid])
    V_s = locally_average_nematic_vectors(V, d2_full, smoothing_std_um)
    V_out[valid] = V_s

    return {"valid_mask": valid, "V": V_out, "COM_um": COM_um}

__all__.append("locally_average_nematic_vectors_from_dataframe")
# %%
# nematic_voxel.py
# -------------------------------------------
# Compute nematic tensor and principal axes from voxel contact data.
# Assumes you have a cell's geometric center `c` and a set of voxel coordinates
# on its apical (or other) contact surface `X`. For spherical/convex cells,
# outward normals at contact voxels are approximated by radial unit vectors
# (x_i - c)/||x_i - c||.
#
# Returns the traceless, symmetric nematic tensor N (3x3), its sorted
# eigenvalues (sigma2 <= sigma3 <= sigma1) and corresponding eigenvectors
# (a2, a3, a1). Eigenvectors are axes (sign-ambiguous). The largest eigenvalue
# and its eigenvector are the nematic order parameter and director, respectively.
#
# Reference form:
#   N = (3/2) * ( <n n^T> - (1/3) I )
#
# where the average <·> is taken over contact voxels with equal weight.
#
# Author: ChatGPT (protocolized for reproducibility)
# -------------------------------------------

@dataclass(frozen=True)
class NematicResult:
    """
    Container for nematic results.

    Attributes
    ----------
    N : (3,3) ndarray
        Traceless, symmetric nematic tensor.
    sigmas : (3,) ndarray
        Sorted eigenvalues (sigma2 <= sigma3 <= sigma1).
    axes : (3,3) ndarray
        Columns are eigenvectors corresponding to `sigmas` in the same order:
        [a2, a3, a1]. Each is a unit vector; sign is arbitrary (axis, not arrow).
    n_valid : int
        Number of valid contact voxels used (r > 0 after center subtraction).
    """
    N: np.ndarray
    sigmas: np.ndarray
    axes: np.ndarray
    n_valid: int


def _outer_mean(unit_vectors: np.ndarray) -> np.ndarray:
    """
    Compute the mean outer product <n n^T> for an array of unit vectors.

    Parameters
    ----------
    unit_vectors : (M, 3) float array
        Each row is a unit vector n_i.

    Returns
    -------
    Q : (3,3) float array
        Mean second-moment tensor Q = (1/M) sum_i n_i n_i^T.
    """
    if unit_vectors.ndim != 2 or unit_vectors.shape[1] != 3:
        raise ValueError("unit_vectors must have shape (M, 3).")
    if unit_vectors.shape[0] == 0:
        raise ValueError("unit_vectors is empty.")
    # Vectorized outer products and mean
    # (M,3,1) * (M,1,3) -> (M,3,3) -> mean over axis 0
    outer = unit_vectors[:, :, None] * unit_vectors[:, None, :]
    Q = outer.mean(axis=0)
    return Q


def _nematic_from_Q(Q: np.ndarray) -> np.ndarray:
    """
    Convert a second-moment tensor Q = <n n^T> into a nematic tensor N.

    N = (3/2) * ( Q - (I/3) )

    Parameters
    ----------
    Q : (3,3) float array

    Returns
    -------
    N : (3,3) float array
    """
    if Q.shape != (3, 3):
        raise ValueError("Q must have shape (3, 3).")
    I = np.eye(3, dtype=Q.dtype)
    N = 1.5 * (Q - I / 3.0)
    # Numerical symmetrization to kill tiny asymmetries
    N = 0.5 * (N + N.T)
    return N


def _eigendecompose_sorted(N: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecompose symmetric tensor and sort ascending by eigenvalue.

    Parameters
    ----------
    N : (3,3) float array, symmetric

    Returns
    -------
    sigmas : (3,) float array
        Sorted eigenvalues (ascending): [sigma2, sigma3, sigma1]
    axes : (3,3) float array
        Columns are eigenvectors corresponding to `sigmas`.
    """
    if N.shape != (3, 3):
        raise ValueError("N must have shape (3, 3).")
    # For symmetric matrices use eigh (guarantees real eigenpairs)
    sigmas, axes = np.linalg.eigh(N)
    # eigh returns ascending already, but be explicit
    order = np.argsort(sigmas)
    sigmas = sigmas[order]
    axes = axes[:, order]
    return sigmas, axes


def radial_normals_from_contacts(
    X_contacts: np.ndarray,
    center: np.ndarray,
    *, eps: float = 0.0
) -> np.ndarray:
    """
    Compute outward unit normals at contact voxels as radial vectors
    from the geometric center: n_i = (x_i - c) / ||x_i - c||.

    Parameters
    ----------
    X_contacts : (M,3) float array
        Coordinates of contact-site voxels (apical or other label).
        Assumed to be boundary voxels; equal weights are used.
    center : (3,) float array
        Geometric center (unweighted mean of cell voxels).
        This is the "center of mass" in your usage.
    eps : float, optional
        Minimum allowed radius ||x_i - c|| to consider a voxel valid.
        Voxels with radius <= eps are discarded to avoid division by zero.
        Default 0.0 (discard exact-center coincidences only).

    Returns
    -------
    n : (K,3) float array
        Outward unit vectors for the K valid voxels (K <= M).
    """
    X_contacts = np.asarray(X_contacts, dtype=float)
    center = np.asarray(center, dtype=float).reshape(3,)
    if X_contacts.ndim != 2 or X_contacts.shape[1] != 3:
        raise ValueError("X_contacts must have shape (M, 3).")

    V = X_contacts - center[None, :]
    r = np.linalg.norm(V, axis=1)
    valid = r > float(eps)
    if not np.any(valid):
        raise ValueError("No valid contact voxels after radius filtering.")
    n = V[valid] / r[valid][:, None]
    return n


def nematic_from_contact_voxels(
    X_contacts: np.ndarray,
    center: np.ndarray,
    *, eps: float = 0.0
) -> NematicResult:
    """
    Compute nematic tensor and principal axes from contact voxel coordinates
    using radial normals (spherical/convex assumption).

    Parameters
    ----------
    X_contacts : (M,3) float array
        Contact-site voxel coordinates on the cell surface (apical, basal, ...).
        Use one-voxel-thick boundary samples to avoid multi-counting.
    center : (3,) float array
        Geometric center (unweighted average of all cell voxels).
    eps : float, optional
        Discard voxels with ||x_i - c|| <= eps to avoid numerical issues.

    Returns
    -------
    NematicResult
        N (3x3), eigenvalues (sigma2 <= sigma3 <= sigma1),
        axes with columns [a2, a3, a1], and count of valid voxels.
    """
    n = radial_normals_from_contacts(X_contacts, center, eps=eps)
    Q = _outer_mean(n)              # <n n^T>
    N = _nematic_from_Q(Q)          # (3/2)(Q - I/3)
    sigmas, axes = _eigendecompose_sorted(N)

    # Package results
    return NematicResult(N=N, sigmas=sigmas, axes=axes, n_valid=n.shape[0])


def extract_axes_sigmas(result: NematicResult) -> Dict[str, np.ndarray]:
    """
    Convenience: unpack vec1, vec2, vec3, sigma1, sigma2, sigma3 in the paper's naming.

    Paper's convention:
      - eigenvalues sorted such that sigma2 <= sigma3 <= sigma1
      - vec1 is the eigenvector for sigma1 (largest)
      - vec2 is the eigenvector for sigma2 (smallest)
      - vec3 is +/- vec1 x vec2 (implicitly the middle one; here we return the middle)
        Note: we also return the computed vec3 as the eigenvector of sigma3.

    Returns
    -------
    dict with keys:
        'vec1', 'vec2', 'vec3', 'sigma1', 'sigma2', 'sigma3'
    """
    sigma2, sigma3, sigma1 = result.sigmas.tolist()
    vec2 = result.axes[:, 0]
    vec3 = result.axes[:, 1]
    vec1 = result.axes[:, 2]
    return {
        "vec1": vec1,
        "vec2": vec2,
        "vec3": vec3,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma3": sigma3,
    }

def calculate_and_extract_nematic_results(
    X_contacts: np.ndarray,
    center: np.ndarray,
    *, eps: float = 0.0
) -> Dict[str, np.ndarray]:
    """
    Convenience: compute nematic from contact voxels and extract axes and sigmas.

    Parameters
    ----------
    X_contacts : (M,3) float array
        Contact-site voxel coordinates on the cell surface (apical, basal, ...).
        Use one-voxel-thick boundary samples to avoid multi-counting.
    center : (3,) float array
        Geometric center (unweighted average of all cell voxels).
    eps : float, optional
        Discard voxels with ||x_i - c|| <= eps to avoid numerical issues.

    Returns
    -------
    dict with keys:
        'N' : (3,3) nematic tensor
        'vec1', 'vec2', 'vec3' : (3,) eigenvectors (axes)
        'sigma1', 'sigma2', 'sigma3' : eigenvalues
        'n_valid' : int, number of valid contact voxels used
    """
    try:
        result = nematic_from_contact_voxels(X_contacts, center, eps=eps)
        axes_sigmas = extract_axes_sigmas(result)
        return {
            "N": result.N,
            **axes_sigmas,
            "n_valid": result.n_valid
        }
    except ValueError as e:
        if str(e) == "No valid contact voxels after radius filtering.":
            # Return NaNs if no valid voxels
            return {
                "N": np.full((3, 3), np.nan),
                "vec1": np.full(3, np.nan),
                "vec2": np.full(3, np.nan),
                "vec3": np.full(3, np.nan),
                "sigma1": np.nan,
                "sigma2": np.nan,
                "sigma3": np.nan,
                "n_valid": 0
            }


# # -------------------------
# # Minimal usage example
# # -------------------------
# if __name__ == "__main__":
#     # Example with a spherical cap: points on z >= z0 on the unit sphere
#     rng = np.random.default_rng(42)
#     n_pts = 5000
#     z0 = 0.3  # cap threshold; larger z0 => smaller cap => stronger anisotropy

#     # Sample directions uniformly on sphere via normal distribution
#     X = rng.normal(size=(n_pts, 3))
#     X /= np.linalg.norm(X, axis=1, keepdims=True)

#     # Keep a spherical cap (proxy for an "apical patch")
#     cap = (X[:, 0] <=-.5) | (X[:, 0] >= 0.5)
#     X_cap = X[cap]

#     # Place the cell center at origin (geometric center)
#     c = np.array([0.0, 0.0, 0.0])

#     # Treat the contact voxels as sitting on the surface (unit radius)
#     # For voxel data, you would pass actual voxel coordinates in physical units.
#     res = nematic_from_contact_voxels(X_cap, c)
#     unpacked = extract_axes_sigmas(res)

#     print("Nematic tensor N:\n", res.N)
#     print("eigenvalues (sigma2 <= sigma3 <= sigma1): ", res.sigmas)
#     print("a1 (director): ", unpacked["a1"])
#     print("a2: ", unpacked["a2"])
#     print("a3: ", unpacked["a3"])
#     print("# valid samples: ", res.n_valid)

#     sigma2, sigma3, sigma1 = res.sigmas  # per your printout ordering
#     plot_cloud_and_axes_plotly(
#         X,
#         cap_mask=cap,
#         a1=unpacked["a1"],
#         a2=unpacked["a2"],
#         sigma1=sigma1,     # cylinder length for a1
#         sigma2=sigma2,     # cylinder length for a2
#         c=c,
#         radius=0.03,       # tweak thickness to taste
#         show_sphere=True
#     )
# # %%
