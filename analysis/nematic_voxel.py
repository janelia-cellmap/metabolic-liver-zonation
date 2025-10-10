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

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Dict
import numpy as np

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
