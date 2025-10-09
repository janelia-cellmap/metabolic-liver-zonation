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
    Convenience: unpack a1, a2, a3, sigma1, sigma2, sigma3 in the paper's naming.

    Paper's convention:
      - eigenvalues sorted such that sigma2 <= sigma3 <= sigma1
      - a1 is the eigenvector for sigma1 (largest)
      - a2 is the eigenvector for sigma2 (smallest)
      - a3 is +/- a1 x a2 (implicitly the middle one; here we return the middle)
        Note: we also return the computed a3 as the eigenvector of sigma3.

    Returns
    -------
    dict with keys:
        'a1', 'a2', 'a3', 'sigma1', 'sigma2', 'sigma3'
    """
    sigma2, sigma3, sigma1 = result.sigmas.tolist()
    a2 = result.axes[:, 0]
    a3 = result.axes[:, 1]
    a1 = result.axes[:, 2]
    return {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma3": sigma3,
    }

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_cloud_and_vectors(
    X: np.ndarray,
    cap_mask: np.ndarray,
    vectors: list,
    c: np.ndarray = None,
    color_all: str = "tab:blue",
    color_cap: str = "tab:orange",
    vector_colors: list = None,
    s_all: int = 8,
    s_cap: int = 12,
    sphere_wireframe: bool = True,
):
    """
    Interactive 3D scatter plot of points on a sphere and one or more vectors.

    Parameters
    ----------
    X : (N,3) array
        Point cloud on the sphere.
    cap_mask : (N,) bool array
        Mask selecting a spherical cap subset of X.
    vectors : list of (3,) arrays
        Direction vectors to plot from center c.
    c : (3,) array, optional
        Center of sphere (default origin).
    color_all, color_cap : str
        Colors for all points and cap points.
    vector_colors : list of str
        Colors for each vector; defaults to distinct matplotlib tab colors.
    s_all, s_cap : int
        Scatter point sizes.
    sphere_wireframe : bool
        If True, draw a light wireframe unit sphere for context.
    """

    if c is None:
        c = np.zeros(3)
    if vector_colors is None:
        vector_colors = ["tab:red", "tab:green", "tab:purple"]

    X_cap = X[cap_mask]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    # Plot all points
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
               s=s_all, alpha=0.35, label="X (all)", c=color_all)

    # Plot cap points
    ax.scatter(X_cap[:, 0], X_cap[:, 1], X_cap[:, 2],
               s=s_cap, alpha=0.9, label="X_cap (cap)", c=color_cap)

    # Optional wireframe sphere
    if sphere_wireframe:
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, rstride=4, cstride=4,
                          linewidth=0.4, alpha=0.25, color="gray")

    # Plot vectors from center
    for i, v in enumerate(vectors):
        v = np.asarray(v, dtype=float).reshape(3)
        color = vector_colors[i % len(vector_colors)]
        ax.quiver(c[0], c[1], c[2],
                  v[0], v[1], v[2],
                  length=1.0, arrow_length_ratio=0.12,
                  linewidth=2.0, color=color, label=f"v{i+1} (+)")
        # optional mirrored arrow
        ax.quiver(c[0], c[1], c[2],
                  -v[0], -v[1], -v[2],
                  length=1.0, arrow_length_ratio=0.12,
                  linewidth=1.5, color=color, alpha=0.6, label=f"v{i+1} (-)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D point cloud with spherical cap and vectors")

    # Equal scaling
    def set_equal_3d(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        radius = 0.6 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - radius, x_middle + radius])
        ax.set_ylim3d([y_middle - radius, y_middle + radius])
        ax.set_zlim3d([z_middle - radius, z_middle + radius])

    set_equal_3d(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.show()

import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go

def plot_cloud_and_axes_plotly(
    X: np.ndarray,
    cap_mask: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    sigma1: float,
    sigma2: float,
    c: np.ndarray = None,
    radius: float = 0.03,
    color_all: str = "rgba(31,119,180,0.35)",
    color_cap: str = "rgba(255,127,14,0.95)",
    color_a1: str = "red",
    color_a2: str = "green",
    point_size_all: float = 2.0,
    point_size_cap: float = 3.0,
    show_sphere: bool = True,
    sphere_opacity: float = 0.12,
    n_theta: int = 48,
    n_s: int = 16,
):
    """
    Plots:
      - all points X (one color),
      - cap points X_cap (another color),
      - a1 and a2 as CYLINDERS centered at c and extending equally
        in +/- directions with half-length = sigma (total length = 2*sigma).

    Cylinder parameterization (simplest & robust):
      P(θ, s) = c + s*u + r*(cosθ*b1 + sinθ*b2),
      where u = a/||a||, b1,b2 form an orthonormal basis perpendicular to u,
      θ in [0, 2π), s in [-sigma, +sigma].
    """

    def _normalize(v):
        v = np.asarray(v, float).reshape(3)
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    def _frame_from_dir(vhat):
        """Return orthonormal (u, b1, b2) given unit vhat as u."""
        u = _normalize(vhat)
        # pick a temp vector not parallel to u
        tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        b1 = np.cross(u, tmp)
        b1 = b1 / (np.linalg.norm(b1) + 1e-12)
        b2 = np.cross(u, b1)
        return u, b1, b2

    def _cylinder_pm(c0, a, sigma, r, n_theta=48, n_s=16):
        """
        Cylinder centered at c0, axis along 'a' (unit direction used),
        extending from s=-sigma to s=+sigma.
        Returns (X, Y, Z) meshes for go.Surface.
        """
        c0 = np.asarray(c0, float).reshape(3)
        u, b1, b2 = _frame_from_dir(a)

        theta = np.linspace(0, 2*np.pi, n_theta)
        s = np.linspace(-sigma, +sigma, n_s)
        Theta, S = np.meshgrid(theta, s)  # (n_s, n_theta)

        # P(θ, s) = c + s*u + r*(cosθ*b1 + sinθ*b2)
        cosT, sinT = np.cos(Theta), np.sin(Theta)
        X = c0[0] + S * u[0] + r * (cosT * b1[0] + sinT * b2[0])
        Y = c0[1] + S * u[1] + r * (cosT * b1[1] + sinT * b2[1])
        Z = c0[2] + S * u[2] + r * (cosT * b1[2] + sinT * b2[2])
        return X, Y, Z

    # defaults / data prep
    if c is None:
        c = np.zeros(3, dtype=float)
    X = np.asarray(X, float)
    X_cap = X[cap_mask]

    # unit directions (display length comes only from sigmas)
    a1_u = _normalize(a1)
    a2_u = _normalize(a2)

    # build traces
    traces = []

    # All points
    traces.append(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode="markers",
        marker=dict(size=point_size_all, color=color_all),
        name="X (all)"
    ))

    # Cap points
    traces.append(go.Scatter3d(
        x=X_cap[:, 0], y=X_cap[:, 1], z=X_cap[:, 2],
        mode="markers",
        marker=dict(size=point_size_cap, color=color_cap),
        name="X_cap"
    ))

    # Optional unit sphere for context
    if show_sphere:
        u = np.linspace(0, 2*np.pi, 80)
        v = np.linspace(0, np.pi, 40)
        uu, vv = np.meshgrid(u, v)
        xs = np.cos(uu) * np.sin(vv)
        ys = np.sin(uu) * np.sin(vv)
        zs = np.cos(vv)
        traces.append(go.Surface(
            x=xs, y=ys, z=zs,
            showscale=False,
            opacity=sphere_opacity,
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            hoverinfo="skip",
            name="unit sphere"
        ))

    # Cylinders for a1 (half-length sigma1) and a2 (half-length sigma2)
    X1, Y1, Z1 = _cylinder_pm(c, a1_u, sigma1, radius, n_theta=n_theta, n_s=n_s)
    X2, Y2, Z2 = _cylinder_pm(c, a2_u, sigma2, radius, n_theta=n_theta, n_s=n_s)

    traces.append(go.Surface(
        x=X1, y=Y1, z=Z1, showscale=False, opacity=0.98,
        colorscale=[[0, color_a1], [1, color_a1]],
        name=f"a1 (total len={2*sigma1:.3f})"
    ))
    traces.append(go.Surface(
        x=X2, y=Y2, z=Z2, showscale=False, opacity=0.98,
        colorscale=[[0, color_a2], [1, color_a2]],
        name=f"a2 (total len={2*sigma2:.3f})"
    ))

    # Optional thin axis lines to emphasize exact endpoints
    def _axis_line(c0, uhat, sigma, color, name):
        p0 = c0 - uhat * sigma
        p1 = c0 + uhat * sigma
        return go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode="lines",
            line=dict(width=5, color=color),
            name=name
        )
    traces.append(_axis_line(c, a1_u, sigma1, color_a1, "a1 axis"))
    traces.append(_axis_line(c, a2_u, sigma2, color_a2, "a2 axis"))

    layout = go.Layout(
        title="Cylindrical axes (a1, a2) with half-length = sigma (total = 2*sigma)",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data"
        ),
        legend=dict(x=1.02, y=1.0)
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()




# -------------------------
# Minimal usage example
# -------------------------
if __name__ == "__main__":
    # Example with a spherical cap: points on z >= z0 on the unit sphere
    rng = np.random.default_rng(42)
    n_pts = 5000
    z0 = 0.3  # cap threshold; larger z0 => smaller cap => stronger anisotropy

    # Sample directions uniformly on sphere via normal distribution
    X = rng.normal(size=(n_pts, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    # Keep a spherical cap (proxy for an "apical patch")
    cap = (X[:, 0] <=-.5) | (X[:, 0] >= 0.5)
    X_cap = X[cap]

    # Place the cell center at origin (geometric center)
    c = np.array([0.0, 0.0, 0.0])

    # Treat the contact voxels as sitting on the surface (unit radius)
    # For voxel data, you would pass actual voxel coordinates in physical units.
    res = nematic_from_contact_voxels(X_cap, c)
    unpacked = extract_axes_sigmas(res)

    print("Nematic tensor N:\n", res.N)
    print("eigenvalues (sigma2 <= sigma3 <= sigma1): ", res.sigmas)
    print("a1 (director): ", unpacked["a1"])
    print("a2: ", unpacked["a2"])
    print("a3: ", unpacked["a3"])
    print("# valid samples: ", res.n_valid)

    sigma2, sigma3, sigma1 = res.sigmas  # per your printout ordering
    plot_cloud_and_axes_plotly(
        X,
        cap_mask=cap,
        a1=unpacked["a1"],
        a2=unpacked["a2"],
        sigma1=sigma1,     # cylinder length for a1
        sigma2=sigma2,     # cylinder length for a2
        c=c,
        radius=0.03,       # tweak thickness to taste
        show_sphere=True
    )
# %%
