
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd
import os
import pandas as pd
from nematic_voxel import (
    _gaussian_weights,
    _principal_axis,
    _nematic_weighted,
    _unit,
    locally_average_nematic_vectors,
    locally_average_nematic_vectors_from_dataframe,
)

def plot_vector_field_2d(
    polarity_csv_path: str,
    cell_csv_path: str,
    grid_size_um: float = 25.0,
    plane: str = 'xz',
    output_dir: str = None,
    dataset_name: str = None,
    arrow_scale: float = 30.0,
    use_weighting: bool = False
):
    """
    Plot 2D vector fields of canaliculi, sinusoid, and polarity vectors.
    
    Parameters:
    -----------
    polarity_csv_path : str
        Path to the polarity vectors CSV file
    cell_csv_path : str
        Path to the cell CSV file with COM coordinates
    grid_size_um : float or None
        Grid size in micrometers for binning (default 25 μm).
        If None or False, plot vectors at each cell's COM without binning.
    plane : str
        Plane to plot: 'xz' (average along y) or 'yz' (average along x)
    output_dir : str
        Directory to save plots (optional)
    dataset_name : str
        Name of dataset for plot titles
    arrow_scale : float
        Scale factor for arrow length
    use_weighting : bool
        If True, weight averages by contact counts (canaliculi/sinusoid) or polarity strength.
        If False, use simple unweighted averages (default: True).
        Note: Only applies when grid_size_um is set (binning is enabled).
    """
    # Load data
    polarity_df = pd.read_csv(polarity_csv_path)
    cell_df = pd.read_csv(cell_csv_path)
    
    # Merge to get COM coordinates
    merged_df = polarity_df.merge(
        cell_df[['Object ID', 'COM Z (nm)', 'COM Y (nm)', 'COM X (nm)']],
        left_on='Cell ID',
        right_on='Object ID',
        how='left'
    )
    
    # Filter out cells without valid polarity
    valid_df = merged_df[merged_df['Polarity Strength'].notna()].copy()
    
    # Determine if binning is enabled
    use_binning = grid_size_um not in [None, False]
    
    # Convert grid size from μm to nm if binning
    if use_binning:
        grid_size_nm = grid_size_um * 1000
    
    # Determine axes based on plane
    if plane.lower() == 'xz':
        # Average along Y, plot X vs Z
        axis1_name, axis2_name, avg_axis_name = 'X', 'Z', 'Y'
        axis1_col = 'COM X (nm)'
        axis2_col = 'COM Z (nm)'
        # Vector components for the plane
        can_vec1, can_vec2 = 'Canaliculi Mean Dir X', 'Canaliculi Mean Dir Z'
        sin_vec1, sin_vec2 = 'Sinusoids Mean Dir X', 'Sinusoids Mean Dir Z'
        pol_vec1, pol_vec2 = 'Polarity Axis X', 'Polarity Axis Z'
    else:  # 'yz'
        # Average along X, plot Y vs Z
        axis1_name, axis2_name, avg_axis_name = 'Y', 'Z', 'X'
        axis1_col = 'COM Y (nm)'
        axis2_col = 'COM Z (nm)'
        can_vec1, can_vec2 = 'Canaliculi Mean Dir Y', 'Canaliculi Mean Dir Z'
        sin_vec1, sin_vec2 = 'Sinusoids Mean Dir Y', 'Sinusoids Mean Dir Z'
        pol_vec1, pol_vec2 = 'Polarity Axis Y', 'Polarity Axis Z'
    
    if use_binning:
        # Create grid bins
        axis1_min = valid_df[axis1_col].min()
        axis1_max = valid_df[axis1_col].max()
        axis2_min = valid_df[axis2_col].min()
        axis2_max = valid_df[axis2_col].max()
        
        axis1_bins = np.arange(axis1_min, axis1_max + grid_size_nm, grid_size_nm)
        axis2_bins = np.arange(axis2_min, axis2_max + grid_size_nm, grid_size_nm)
        
        # Assign grid cells
        valid_df['grid_axis1'] = pd.cut(valid_df[axis1_col], bins=axis1_bins, labels=False)
        valid_df['grid_axis2'] = pd.cut(valid_df[axis2_col], bins=axis2_bins, labels=False)
        
        # Prepare grid for storing averaged vectors
        n_axis1 = len(axis1_bins) - 1
        n_axis2 = len(axis2_bins) - 1
        
        grid_axis1_centers = (axis1_bins[:-1] + axis1_bins[1:]) / 2 / 1000  # Convert to μm
        grid_axis2_centers = (axis2_bins[:-1] + axis2_bins[1:]) / 2 / 1000  # Convert to μm
        
        # Initialize storage for vectors
        can_vec_field_1 = np.full((n_axis2, n_axis1), np.nan)
        can_vec_field_2 = np.full((n_axis2, n_axis1), np.nan)
        sin_vec_field_1 = np.full((n_axis2, n_axis1), np.nan)
        sin_vec_field_2 = np.full((n_axis2, n_axis1), np.nan)
        pol_vec_field_1 = np.full((n_axis2, n_axis1), np.nan)
        pol_vec_field_2 = np.full((n_axis2, n_axis1), np.nan)
        cell_counts = np.zeros((n_axis2, n_axis1), dtype=int)
        
        # Calculate averaged vectors for each grid cell
        for i in range(n_axis1):
            for j in range(n_axis2):
                grid_data = valid_df[(valid_df['grid_axis1'] == i) & (valid_df['grid_axis2'] == j)]
                
                if len(grid_data) == 0:
                    continue
                
                cell_counts[j, i] = len(grid_data)
                
                # Canaliculi vectors - weighted or unweighted
                can_mask = grid_data[can_vec1].notna() & grid_data[can_vec2].notna()
                if can_mask.sum() > 0:
                    if use_weighting:
                        # Weighted average by number of canaliculi contact voxels
                        can_weights = grid_data.loc[can_mask, 'Num Canaliculi Contact Voxels'].values
                        if can_weights.sum() > 0:
                            can_vec_field_1[j, i] = np.average(grid_data.loc[can_mask, can_vec1], weights=can_weights)
                            can_vec_field_2[j, i] = np.average(grid_data.loc[can_mask, can_vec2], weights=can_weights)
                        else:
                            can_vec_field_1[j, i] = grid_data.loc[can_mask, can_vec1].mean()
                            can_vec_field_2[j, i] = grid_data.loc[can_mask, can_vec2].mean()
                    else:
                        # Simple unweighted average
                        can_vec_field_1[j, i] = grid_data.loc[can_mask, can_vec1].mean()
                        can_vec_field_2[j, i] = grid_data.loc[can_mask, can_vec2].mean()
                # else: leave as NaN
                
                # Sinusoid vectors - weighted or unweighted
                sin_mask = grid_data[sin_vec1].notna() & grid_data[sin_vec2].notna()
                if sin_mask.sum() > 0:
                    if use_weighting:
                        # Weighted average by number of sinusoid contact voxels
                        sin_weights = grid_data.loc[sin_mask, 'Num Sinusoid Contact Voxels'].values
                        if sin_weights.sum() > 0:
                            sin_vec_field_1[j, i] = np.average(grid_data.loc[sin_mask, sin_vec1], weights=sin_weights)
                            sin_vec_field_2[j, i] = np.average(grid_data.loc[sin_mask, sin_vec2], weights=sin_weights)
                        else:
                            sin_vec_field_1[j, i] = grid_data.loc[sin_mask, sin_vec1].mean()
                            sin_vec_field_2[j, i] = grid_data.loc[sin_mask, sin_vec2].mean()
                    else:
                        # Simple unweighted average
                        sin_vec_field_1[j, i] = grid_data.loc[sin_mask, sin_vec1].mean()
                        sin_vec_field_2[j, i] = grid_data.loc[sin_mask, sin_vec2].mean()
                # else: leave as NaN
                
                # Polarity vectors - weighted or unweighted
                pol_mask = grid_data[pol_vec1].notna() & grid_data[pol_vec2].notna() & grid_data['Polarity Strength'].notna()
                if pol_mask.sum() > 0:
                    if use_weighting:
                        # Weighted average by polarity strength
                        pol_weights = grid_data.loc[pol_mask, 'Polarity Strength'].values
                        if pol_weights.sum() > 0:
                            pol_vec_field_1[j, i] = np.average(grid_data.loc[pol_mask, pol_vec1], weights=pol_weights)
                            pol_vec_field_2[j, i] = np.average(grid_data.loc[pol_mask, pol_vec2], weights=pol_weights)
                        else:
                            pol_vec_field_1[j, i] = grid_data.loc[pol_mask, pol_vec1].mean()
                            pol_vec_field_2[j, i] = grid_data.loc[pol_mask, pol_vec2].mean()
                    else:
                        # Simple unweighted average
                        pol_vec_field_1[j, i] = grid_data.loc[pol_mask, pol_vec1].mean()
                        pol_vec_field_2[j, i] = grid_data.loc[pol_mask, pol_vec2].mean()
                # else: leave as NaN
        
        # Create meshgrid for plotting
        X, Y = np.meshgrid(grid_axis1_centers, grid_axis2_centers)
    else:
        # No binning: use cell COM positions directly
        # Extract positions in μm
        X = valid_df[axis1_col].values / 1000  # Convert nm to μm
        Y = valid_df[axis2_col].values / 1000
        
        # Extract vector components
        can_vec_field_1 = valid_df[can_vec1].values
        can_vec_field_2 = valid_df[can_vec2].values
        sin_vec_field_1 = valid_df[sin_vec1].values
        sin_vec_field_2 = valid_df[sin_vec2].values
        pol_vec_field_1 = valid_df[pol_vec1].values
        pol_vec_field_2 = valid_df[pol_vec2].values
    
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Determine subtitle based on weighting (only applies when binning is enabled)
    if use_binning:
        weight_subtitle = "(weighted by contact voxel count)" if use_weighting else "(unweighted average)"
        pol_weight_subtitle = "(weighted by polarity strength)" if use_weighting else "(unweighted average)"
        avg_text = f", averaged along {avg_axis_name}"
    else:
        weight_subtitle = ""
        pol_weight_subtitle = ""
        avg_text = " at cell COM"
    
    # Plot 1: Canaliculi vectors (green)
    ax = axes[0]
    mask = ~np.isnan(can_vec_field_1)
    # Note: scale is INVERSE to arrow length. Lower scale = longer arrows.
    # For unit vectors (~0-1 magnitude), scale=10-50 works well.
    # Alternatively, use scale_units='xy' for better control
    ax.quiver(X[mask], Y[mask], can_vec_field_1[mask], can_vec_field_2[mask],
              color='green', scale=arrow_scale, scale_units='xy', angles='xy', 
              width=0.003, alpha=0.7)
    ax.set_xlabel(f'{axis1_name} position (μm)', fontsize=12)
    ax.set_ylabel(f'{axis2_name} position (μm)', fontsize=12)
    ax.set_title(f'Canaliculi Mean Direction Vectors\n{weight_subtitle}{avg_text}', fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Sinusoid vectors (red)
    ax = axes[1]
    mask = ~np.isnan(sin_vec_field_1)
    ax.quiver(X[mask], Y[mask], sin_vec_field_1[mask], sin_vec_field_2[mask],
              color='red', scale=arrow_scale, scale_units='xy', angles='xy',
              width=0.003, alpha=0.7)
    ax.set_xlabel(f'{axis1_name} position (μm)', fontsize=12)
    ax.set_ylabel(f'{axis2_name} position (μm)', fontsize=12)
    ax.set_title(f'Sinusoid Mean Direction Vectors\n{weight_subtitle}{avg_text}', fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Polarity vectors (blue)
    ax = axes[2]
    mask = ~np.isnan(pol_vec_field_1)
    ax.quiver(X[mask], Y[mask], pol_vec_field_1[mask], pol_vec_field_2[mask],
              color='blue', scale=arrow_scale, scale_units='xy', angles='xy',
              width=0.003, alpha=0.7)
    ax.set_xlabel(f'{axis1_name} position (μm)', fontsize=12)
    ax.set_ylabel(f'{axis2_name} position (μm)', fontsize=12)
    ax.set_title(f'Polarity Vectors\n{pol_weight_subtitle}{avg_text}', fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    # Create main title based on binning mode
    if use_binning:
        main_title = f'Vector Field Analysis - {dataset_name} ({plane.upper()} plane, {grid_size_um}μm grid)' if dataset_name else f'Vector Field Analysis ({plane.upper()} plane, {grid_size_um}μm grid)'
    else:
        main_title = f'Vector Field Analysis - {dataset_name} ({plane.upper()} plane, per-cell)' if dataset_name else f'Vector Field Analysis ({plane.upper()} plane, per-cell)'
    
    plt.suptitle(main_title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if use_binning:
            weight_tag = "weighted" if use_weighting else "unweighted"
            output_file = os.path.join(output_dir, f'vector_field_{plane}_{weight_tag}_{dataset_name}.png')
        else:
            output_file = os.path.join(output_dir, f'vector_field_{plane}_percell_{dataset_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")
    
    plt.show()
    
    if use_binning:
        return {
            'grid_axis1_centers': grid_axis1_centers,
            'grid_axis2_centers': grid_axis2_centers,
            'canaliculi_vectors': (can_vec_field_1, can_vec_field_2),
            'sinusoid_vectors': (sin_vec_field_1, sin_vec_field_2),
            'polarity_vectors': (pol_vec_field_1, pol_vec_field_2),
            'cell_counts': cell_counts
        }
    else:
        return {
            'positions': (X, Y),
            'canaliculi_vectors': (can_vec_field_1, can_vec_field_2),
            'sinusoid_vectors': (sin_vec_field_1, sin_vec_field_2),
            'polarity_vectors': (pol_vec_field_1, pol_vec_field_2)
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

def plot_projected_points_with_two_vectors(
    X: np.ndarray,
    c: np.ndarray,
    vec1: np.ndarray,
    vec2: np.ndarray,
    sigma1: float,
    sigma2: float,
    # vector scaling
    vector_scale: float = 1.0,          # global multiplier applied to both sigmas
    symmetric_vectors: bool = False,    # False: ray c->(c+scaled vec); True: line through c
    # visuals
    show_original_points: bool = False,
    show_rays: bool = False,            # rays from c to projected points
    sphere_opacity: float = 0.18,
    point_size_proj: float = 4.0,
    point_size_orig: float = 2.0,
    color_proj: str = "rgba(255,127,14,0.95)",
    color_orig: str = "rgba(31,119,180,0.35)",
    color_rays: str = "rgba(0,0,0,0.25)",
    color_sphere: str = "lightgray",
    color_vec1: str = "red",
    color_vec2: str = "green",
    # sphere mesh quality
    n_theta: int = 96,
    n_phi: int = 48,
    title: str = "Unit-sphere projection with vec1/vec2",
    show: bool = False,                 # avoid double-render in notebooks; call fig.show() yourself if desired
):
    """
    Projects points X onto the unit sphere centered at c and plots:
      - transparent sphere mesh
      - projected points
      - two vectors (vec1, vec2) named 'vec1' and 'vec2'

    Vector scaling:
      Let s1 = vector_scale * sigma1, s2 = vector_scale * sigma2.
      If symmetric_vectors=False: draw rays c -> c + s1*vec1 and c -> c + s2*vec2 (sign flips direction).
      If symmetric_vectors=True: draw lines through c with half-length |s|*||vec|| along ±(vec/||vec||).
    """
    # --- prep & projection ---
    X = np.asarray(X, float).reshape(-1, 3)
    c = np.asarray(c, float).reshape(3)

    V = X - c
    norms = np.linalg.norm(V, axis=1)
    good = norms > 1e-12

    U = np.zeros_like(V)
    U[good] = V[good] / norms[good, None]  # unit directions
    X_proj = c + U                          # unit sphere projection (R=1)

    # --- sphere mesh ---
    u = np.linspace(0, 2*np.pi, n_theta)
    v = np.linspace(0, np.pi, n_phi)
    uu, vv = np.meshgrid(u, v)
    xs = c[0] + np.cos(uu) * np.sin(vv)
    ys = c[1] + np.sin(uu) * np.sin(vv)
    zs = c[2] + np.cos(vv)

    traces = [go.Surface(
        x=xs, y=ys, z=zs,
        showscale=False,
        opacity=sphere_opacity,
        colorscale=[[0, color_sphere], [1, color_sphere]],
        hoverinfo="skip",
        name="unit sphere"
    )]

    # --- optional original points ---
    if show_original_points and np.any(good):
        traces.append(go.Scatter3d(
            x=X[good, 0], y=X[good, 1], z=X[good, 2],
            mode="markers",
            marker=dict(size=point_size_orig, color=color_orig),
            name="original points"
        ))

    # --- projected points ---
    if np.any(good):
        traces.append(go.Scatter3d(
            x=X_proj[good, 0], y=X_proj[good, 1], z=X_proj[good, 2],
            mode="markers",
            marker=dict(size=point_size_proj, color=color_proj),
            name="projected points"
        ))

    # --- optional rays from center to projected points ---
    if show_rays and np.any(good):
        x_lines, y_lines, z_lines = [], [], []
        for p in X_proj[good]:
            x_lines += [c[0], p[0], np.nan]
            y_lines += [c[1], p[1], np.nan]
            z_lines += [c[2], p[2], np.nan]
        traces.append(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode="lines",
            line=dict(width=2, color=color_rays),
            name="rays (c→proj)"
        ))

    # --- two vectors: vec1, vec2 ---
    vec1 = np.asarray(vec1, float).reshape(3)
    vec2 = np.asarray(vec2, float).reshape(3)
    s1 = float(vector_scale) * float(sigma1)
    s2 = float(vector_scale) * float(sigma2)

    def _add_vector(vec, s, color, name):
        n = np.linalg.norm(vec)
        if n < 1e-12:
            return
        if symmetric_vectors:
            # line through c with half-length = |s| * ||vec||
            half_len = abs(s) * n
            uhat = vec / n
            p0 = c - uhat * half_len
            p1 = c + uhat * half_len
        else:
            # one-sided ray: c -> c + s*vec  (sign flips direction)
            p0 = c
            p1 = c + s * vec
        traces.append(go.Scatter3d(
            x=[p0[0], p1[0]],
            y=[p0[1], p1[1]],
            z=[p0[2], p1[2]],
            mode="lines",
            line=dict(width=6, color=color),
            name=name
        ))

    _add_vector(vec1, s1, color_vec1, "vec1")
    _add_vector(vec2, s2, color_vec2, "vec2")

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data"
        ),
        legend=dict(x=1.02, y=1.0)
    )

    fig = go.Figure(data=traces, layout=layout)
    if show:
        fig.show()
    return fig



# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig

# ----------------- helpers -----------------

def _unit(v, eps=1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def _plane_indices(plane: str):
    plane = plane.upper()
    if plane == "XY": return 0, 1
    if plane == "XZ": return 0, 2
    if plane == "YZ": return 1, 2
    raise ValueError("plane must be 'XY', 'XZ', or 'YZ'")

def _columns_for_base(df: pd.DataFrame, base: str):
    """Return column names (Xcol, Ycol, Zcol) for a base like 'Canaliculi a1'."""
    def pick(suffix):
        target = f"{base} {suffix}"
        # exact match first
        for c in df.columns:
            if c == target:
                return c
        # case-insensitive fallback
        target_low = target.lower()
        for c in df.columns:
            if c.lower() == target_low:
                return c
        raise KeyError(f"Missing column '{target}'")
    return pick("X"), pick("Y"), pick("Z")

def _get_com_columns(df: pd.DataFrame):
    for triplet in [
        ("COM X (nm)", "COM Y (nm)", "COM Z (nm)"),
        ("COM X", "COM Y", "COM Z"),
        ("Center X", "Center Y", "Center Z"),
    ]:
        if all(c in df.columns for c in triplet):
            return triplet
    raise KeyError("Could not find COM columns (try 'COM X (nm)', 'COM Y (nm)', 'COM Z (nm)').")

def _convert_units(arr, from_units: str):
    fu = (from_units or "").lower()
    if fu in ("um","µm","micron","microns"): return arr.astype(float), 1.0
    if fu in ("nm","nanometer","nanometers"): return arr.astype(float)*1e-3, 1e-3
    if fu in ("mm","millimeter","millimeters"): return arr.astype(float)*1e3, 1e3
    return arr.astype(float), 1.0  # assume already µm

# ----------------- main function -----------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_projected_axes_from_csv(
    csv_path: str,
    bases: list,                     # e.g., ["Canaliculi a1"] or multiple
    plane: str = "XZ",               # "XY" | "XZ" | "YZ" (used for 2D projection AND 3D camera)
    length_3d: float = 1.0,          # half-length in 3D BEFORE projection (same units as COM)
    com_units: str = "nm",
    smoothing_std_um: float = None,  # e.g., 20.0 for local averaging in µm
    sigma_column: str = None,        # optional: name of per-row sigma column that must be finite to count as valid
    colors_for_bases: dict = None,   # optional mapping base->color for line segments
    figsize=(7,6),
    com_size=10,
    com_alpha=0.8,

    # --- 3D options ---
    three_d: bool = True,            # default True: interactive 3D (no projection)
    show_invalid_in_3d: bool = True, # show invalid COMs (red) in 3D mode
    view_plane: str = None,          # if None, uses `plane`; else "XY"|"XZ"|"YZ"
    mesh_path: str = None,           # directory containing 1.ply and/or 2.ply
    mesh1_color: str = "rgba(31,119,180,0.45)",
    mesh2_color: str = "rgba(255,127,14,0.45)",
    mesh_opacity: float = None,      # if None, derive from rgba; else override
    horizontal: bool = True,         # NEW: rotate so Z runs along the Plotly X-axis
    fig_width: int = 1200,           # bigger canvas
    fig_height: int = 900
):
    """
    Modes
    -----
    - 3D (three_d=True, default): draw COMs + per-base symmetric ± line segments in 3D (no projection).
      If mesh_path is provided, overlay mesh_path/1.ply and mesh_path/2.ply as semi-transparent meshes.
      If horizontal=True, rotate coordinates so original Z -> Plotly X (scene lies horizontally).
      Camera is oriented to the requested plane.

    - 2D (three_d=False): original behavior. Segments are built in 3D, then projected to the chosen plane.

    Validity
    --------
    A row is "valid" if all requested base components are finite and (if provided) sigma_column is finite.

    Returns
    -------
    - If three_d=False: (matplotlib_figure, matplotlib_axes)
    - If three_d=True:  plotly_figure
    """

    # ---------- dependency: your helpers (expected to exist) ----------
    # _plane_indices(plane) -> (px, py)
    # _get_com_columns(df) -> (col_x, col_y, col_z)
    # _convert_units(COM, com_units) -> (COM_um, factor)
    # _columns_for_base(df, base) -> (cx, cy, cz)
    # _unit(V) -> row-wise normalized vectors
    # _gaussian_weights(d2_row, sigma_um) -> weights
    # _nematic_weighted(V, w) -> 3x3 structure tensor
    # _principal_axis(N) -> principal unit vector from tensor

    # ---------- small internals ----------
    import re
    def _camera_for_plane(p):
        p = (p or "XZ").upper()
        if p == "XY":   # look along +Z
            return dict(eye=dict(x=0, y=0, z=2.8), up=dict(x=0, y=1, z=0))
        if p == "XZ":   # look along +Y
            return dict(eye=dict(x=0, y=2.8, z=0.0001), up=dict(x=0, y=0, z=1))
        if p == "YZ":   # look along +X
            return dict(eye=dict(x=2.8, y=0, z=0.0001), up=dict(x=0, y=0, z=1))
        return dict(eye=dict(x=1.9, y=1.9, z=1.9), up=dict(x=0, y=0, z=1))

    def _parse_rgba(rgba_str, default_opacity=0.35):
        if isinstance(rgba_str, str) and rgba_str.lower().startswith("rgba"):
            m = re.findall(r"[\d.]+", rgba_str)
            if len(m) == 4:
                r, g, b, a = map(float, m)
                return f"rgb({int(r)},{int(g)},{int(b)})", float(a)
        return rgba_str, float(default_opacity)

    def _load_mesh_trace(path, name, color_rgba, opacity_override=None):
        """Return a Plotly Mesh3d trace or None. Requires trimesh if available."""
        try:
            import trimesh
            m = trimesh.load(path, force='mesh')
            V = np.asarray(m.vertices)
            F = np.asarray(m.faces)
            if V.size == 0 or F.size == 0:
                return None
            rgb, alpha = _parse_rgba(color_rgba)
            op = alpha if opacity_override is None else float(opacity_override)
            return go.Mesh3d(
                x=V[:,0], y=V[:,1], z=V[:,2],
                i=F[:,0], j=F[:,1], k=F[:,2],
                color=rgb, opacity=op, flatshading=True, name=name
            )
        except Exception:
            return None

    def _swap_for_horizontal(x, y, z):
        """Map original (X,Y,Z) -> (Xh, Yh, Zh) so Z runs horizontally along Plotly X."""
        # Choose (Xh, Yh, Zh) = (Z, Y, -X) for right-handed feel with 'up' = +Z.
        return z, y, -x

    # ---------- load + prep ----------
    df = pd.read_csv(csv_path)
    px, py = _plane_indices(plane if plane is not None else "XZ")
    axis_labels = ["X","Y","Z"]

    # COMs
    com_x, com_y, com_z = _get_com_columns(df)
    COM = df[[com_x, com_y, com_z]].to_numpy(float)  # (N,3)
    COM_plot2D = COM[:, [px, py]]

    # COM in µm (for smoothing geometry)
    COM_um, _ = _convert_units(COM, com_units)

    # validity
    valid = np.ones(len(df), dtype=bool)
    base_cols = {}
    for base in bases:
        cx, cy, cz = _columns_for_base(df, base)
        base_cols[base] = (cx, cy, cz)
        V = df[[cx, cy, cz]].to_numpy(float)
        valid &= np.all(np.isfinite(V), axis=1)

    if sigma_column is not None:
        if sigma_column not in df.columns:
            raise KeyError(f"Specified sigma_column '{sigma_column}' not found in CSV.")
        valid &= np.isfinite(df[sigma_column].to_numpy(float))

    # smoothing distances (among valid only)
    if smoothing_std_um is not None and smoothing_std_um > 0:
        Pv = COM_um[valid]  # (Nv,3)
        d2_full = np.sum((Pv[:, None, :] - Pv[None, :, :])**2, axis=-1)  # (Nv,Nv)
    else:
        d2_full = None

    # colors
    if colors_for_bases is None:
        default_colors = ["tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
        colors_for_bases = {b: default_colors[i % len(default_colors)] for i, b in enumerate(bases)}

    # =====================================================================
    # 3D MODE (interactive Plotly): no projection; draw symmetric ± lines
    # =====================================================================
    if three_d:
        traces = []

        # COMs
        if np.any(~valid) and show_invalid_in_3d:
            x, y, z = COM[~valid,0], COM[~valid,1], COM[~valid,2]
            if horizontal:
                x, y, z = _swap_for_horizontal(x, y, z)
            traces.append(go.Scatter3d(
                x=x, y=y, z=z, mode="markers",
                marker=dict(size=max(2, com_size//2), color="red", opacity=com_alpha),
                name="COM (invalid)"
            ))
        if np.any(valid):
            x, y, z = COM[valid,0], COM[valid,1], COM[valid,2]
            if horizontal:
                x, y, z = _swap_for_horizontal(x, y, z)
            traces.append(go.Scatter3d(
                x=x, y=y, z=z, mode="markers",
                marker=dict(size=max(2, com_size//2), color="blue", opacity=com_alpha),
                name="COM (valid)"
            ))

        # For each base: smooth (optional), then draw ± segments: C ± length_3d * v̂
        for base in bases:
            cx, cy, cz = base_cols[base]
            V_all = df[[cx, cy, cz]].to_numpy(float)
            V = V_all[valid]
            V = _unit(V)  # (Nv,3)
            C = COM[valid]

            if d2_full is not None:
                V = locally_average_nematic_vectors(V, d2_full, smoothing_std_um)


            P_minus_3D = C - length_3d * V
            P_plus_3D  = C + length_3d * V

            # Build a single segment trace with NaN breaks (fast)
            xs, ys, zs = [], [], []
            for p0, p1 in zip(P_minus_3D, P_plus_3D):
                xs += [p0[0], p1[0], np.nan]
                ys += [p0[1], p1[1], np.nan]
                zs += [p0[2], p1[2], np.nan]

            if horizontal:
                xs, ys, zs = _swap_for_horizontal(np.array(xs), np.array(ys), np.array(zs))

            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(width=3, color=colors_for_bases[base]),
                name=base
            ))

        # Optional mesh overlays
        if mesh_path:
            p1 = os.path.join(mesh_path, "1.ply")
            p2 = os.path.join(mesh_path, "2.ply")
            for pth, nm, col in [(p1, "mesh 1", mesh1_color), (p2, "mesh 2", mesh2_color)]:
                if os.path.isfile(pth):
                    t = _load_mesh_trace(pth, nm, col, mesh_opacity)
                    if t:
                        if horizontal:
                            # rotate vertices by swapping coordinates
                            Xh, Yh, Zh = _swap_for_horizontal(np.array(t.x), np.array(t.y), np.array(t.z))
                            t.x, t.y, t.z = Xh, Yh, Zh
                        traces.append(t)

        # Camera + labels
        cam = _camera_for_plane(view_plane if view_plane else plane)
        if horizontal:
            x_title, y_title, z_title = f"Z (→ X)", "Y", "−X (→ Z)"
        else:
            x_title, y_title, z_title = "X", "Y", "Z"

        layout = go.Layout(
            title=f"3D axes from {os.path.basename(csv_path)} (segments ±{length_3d:.1f} {com_units})",
            scene=dict(
                xaxis=dict(title=x_title, showspikes=False),
                yaxis=dict(title=y_title, showspikes=False),
                zaxis=dict(title=z_title, showspikes=False),
                aspectmode="data",
                camera=cam,
            ),
            margin=dict(l=10, r=10, b=10, t=60),  # tighter margins to fill space
            legend=dict(x=1.02, y=1.0)
        )
        fig = go.Figure(data=traces, layout=layout)
        fig.update_layout(width=fig_width, height=fig_height)
        fig.show()
        return fig

    # =====================================================================
    # 2D MODE (Matplotlib): original behavior (no rotation needed)
    # =====================================================================
    fig, ax = plt.subplots(figsize=figsize)

    # scatter COMs: blue=valid, red=invalid
    ax.scatter(
        COM_plot2D[~valid, 0], COM_plot2D[~valid, 1],
        c="red", s=com_size, alpha=com_alpha, label="COM (invalid)"
    )
    ax.scatter(
        COM_plot2D[valid, 0], COM_plot2D[valid, 1],
        c="blue", s=com_size, alpha=com_alpha, label="COM (valid)"
    )

    # per-base plotting (same math; then project)
    for base in bases:
        cx, cy, cz = base_cols[base]
        V_all = df[[cx, cy, cz]].to_numpy(float)
        V = V_all[valid]
        V = _unit(V)
        C = COM[valid]

        if d2_full is not None:
            V = locally_average_nematic_vectors(V, d2_full, smoothing_std_um)

        Pm3 = C - length_3d * V
        Pp3 = C + length_3d * V
        Pm2 = Pm3[:, [px, py]]
        Pp2 = Pp3[:, [px, py]]

        color = colors_for_bases[base]
        for p0, p1 in zip(Pm2, Pp2):
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=1.8)

        if len(Pm2) > 0:
            ax.plot([Pm2[0,0], Pp2[0,0]], [Pm2[0,1], Pp2[0,1]],
                    color=color, linewidth=2.5, label=base)

    ax.set_aspect("equal", adjustable="box")
    axis_labels = ["X","Y","Z"]
    ax.set_xlabel(axis_labels[px] + (f" ({com_units})" if com_units else ""))
    ax.set_ylabel(axis_labels[py] + (f" ({com_units})" if com_units else ""))
    title = f"Projected {bases} on {plane} (segments built in 3D, half-length={length_3d} {com_units})"
    if smoothing_std_um and smoothing_std_um > 0:
        title += f"  |  smoothing σ={smoothing_std_um} µm"
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    return fig, ax

#%% Mollweide projection of spherical data
import numpy as np
import matplotlib.pyplot as plt

# ---------- geometry helpers ----------
def _normalize(v, eps=1e-12):
    v = np.asarray(v, float).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0, 0.0
    return v / n, n

def _orthonormal_frame(a1, b1):
    """
    Build a right-handed orthonormal basis tied to the cell:
      ẑ := b1̂  (north/south)
      x̂ := normalized projection of a1 onto the plane ⟂ ẑ  (zero meridian direction on equator)
      ŷ := ẑ × x̂
    If a1 ~ parallel to b1, pick a stable fallback for x̂.
    """
    zhat, nz = _normalize(b1)
    if nz == 0:
        raise ValueError("b1 must be nonzero to define the N–S axis")

    # remove b1 component from a1
    a1 = np.asarray(a1, float).reshape(3)
    a1_par = np.dot(a1, zhat) * zhat
    x_try = a1 - a1_par
    nx = np.linalg.norm(x_try)

    if nx < 1e-8:
        # a1 is (near) parallel to b1; choose any vector ⟂ zhat deterministically
        tmp = np.array([1.0, 0.0, 0.0]) if abs(zhat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_try = np.cross(zhat, tmp)
        nx = np.linalg.norm(x_try)

    xhat = x_try / (nx + 1e-12)
    yhat = np.cross(zhat, xhat)
    yhat, _ = _normalize(yhat)

    # re-orthogonalize xhat just in case
    xhat = np.cross(yhat, zhat)
    xhat, _ = _normalize(xhat)

    return xhat, yhat, zhat  # (prime meridian dir on equator, eastward, north)

def _project_to_unit_sphere(P, c):
    """Radially project 3D points to unit sphere centered at c."""
    P = np.asarray(P, float).reshape(-1, 3)
    c = np.asarray(c, float).reshape(3)
    V = P - c
    r = np.linalg.norm(V, axis=1)
    good = r > 1e-12
    U = np.zeros_like(V)
    U[good] = V[good] / r[good, None]
    return U, good

def _spherical_coords_in_frame(U, xhat, yhat, zhat):
    """
    Convert unit vectors U to (lon λ, lat φ) in radians
    for the cell-centric frame (x̂=0° meridian, ŷ=+90°E, ẑ=north).
    """
    U = np.asarray(U, float).reshape(-1, 3)
    # latitude via dot with north
    phi = np.arcsin(np.clip(U @ zhat, -1.0, 1.0))            # φ ∈ [-π/2, π/2]
    # longitude via atan2 in the equatorial plane (x=cosλ, y=sinλ)
    x = U @ xhat
    y = U @ yhat
    lam = np.arctan2(y, x)                                   # λ ∈ (-π, π]
    return lam, phi

# ---------- Mollweide projection ----------
def _mollweide_forward(lam, phi, tol=1e-12, max_iter=16):
    """
    Forward Mollweide projection for arrays of longitudes (λ) and latitudes (φ).
    Returns (x,y). λ, φ in radians, λ ∈ [-π, π].
    Equations:
      Solve for θ: 2θ + sin(2θ) = π sin φ
      x = (2√2/π) * λ * cos θ
      y = √2 * sin θ
    """
    lam = np.asarray(lam, float)
    phi = np.asarray(phi, float)
    # initial guess: θ0 = φ
    theta = phi.copy()

    # Newton iterations on f(θ) = 2θ + sin(2θ) - π sin φ
    for _ in range(max_iter):
        two_theta = 2.0 * theta
        f = two_theta + np.sin(two_theta) - np.pi * np.sin(phi)
        df = 2.0 + 2.0 * np.cos(two_theta)  # derivative wrt θ
        step = f / (df + 1e-16)
        theta_new = theta - step
        if np.max(np.abs(step)) < tol:
            theta = theta_new
            break
        theta = theta_new

    # map to plane
    const = 2.0 * np.sqrt(2.0) / np.pi
    x = const * lam * np.cos(theta)
    y = np.sqrt(2.0) * np.sin(theta)
    return x, y

# ---------- main plotting function ----------
def plot_mollweide_polarity(
    a1, b1, c,
    can_pts=None,     # apical points (N_a,3) in world coords
    sin_pts=None,     # basal  points (N_b,3) in world coords
    # styling
    figsize=(7.5, 4.5),
    show_graticule=True,
    grid_step_deg=30,      # graticule every 30°
    can_style=dict(marker='o', ms=4, lw=0, alpha=0.9, color='#d95f02', label='apical (can)'),
    sin_style=dict(marker='o', ms=4, lw=0, alpha=0.9, color='#1f77b4', label='basal (sin)'),
    pole_style=dict(marker='o', ms=6, color='k', lw=0, zorder=5),
    meridian_color='#444',
    equator_color='#444',
    outline_color='#222',
    title="Mollweide projection (b1 = N–S, a1 = 0° meridian)"
):
    """
    Make a Mollweide map where:
      - b1 defines the north/south axis
      - a1 defines the zero meridian (vertical centerline)
      - Points are first radially projected onto the unit sphere centered at c
      - Both apical (can_pts) and basal (sin_pts) can be plotted if provided
    """
    a1 = np.asarray(a1, float).reshape(3)
    b1 = np.asarray(b1, float).reshape(3)
    c  = np.asarray(c,  float).reshape(3)

    # cell-centric orthonormal frame
    xhat, yhat, zhat = _orthonormal_frame(a1, b1)

    # project + convert to lon/lat
    def points_to_xy(P):
        if P is None:
            return np.empty((0,)), np.empty((0,))
        U, good = _project_to_unit_sphere(P, c)
        U = U[good]
        if U.size == 0:
            return np.empty((0,)), np.empty((0,))
        lam, phi = _spherical_coords_in_frame(U, xhat, yhat, zhat)
        return _mollweide_forward(lam, phi)

    x_can, y_can = points_to_xy(can_pts)
    x_sin, y_sin = points_to_xy(sin_pts)

    # figure + axes
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)

    # outline ellipse of Mollweide (x in [-2√2, 2√2], y in [-√2, √2])
    R_x = 2.0 * np.sqrt(2.0)
    R_y = np.sqrt(2.0)
    tt = np.linspace(0, 2*np.pi, 600)
    outline_x = R_x * np.cos(tt)
    outline_y = R_y * np.sin(tt)
    ax.plot(outline_x, outline_y, color=outline_color, lw=1.0)

    # graticule (optional)
    if show_graticule:
        # meridians (λ constant)
        for lam_deg in range(-150, 181, grid_step_deg):
            lam = np.deg2rad(lam_deg)
            phis = np.linspace(-np.pi/2, np.pi/2, 241)
            xs, ys = _mollweide_forward(lam * np.ones_like(phis), phis)
            ax.plot(xs, ys, color='#cccccc', lw=0.7, alpha=0.8, zorder=0)
            if lam_deg % 60 == 0 and lam_deg != 0:
                ax.text(xs[len(xs)//2], ys[len(ys)//2], f'{lam_deg}°', fontsize=8, ha='center', va='center', color='#888')

        # parallels (φ constant)
        for phi_deg in range(-60, 61, grid_step_deg):
            phi = np.deg2rad(phi_deg)
            lams = np.linspace(-np.pi, np.pi, 361)
            xs, ys = _mollweide_forward(lams, phi * np.ones_like(lams))
            ax.plot(xs, ys, color='#cccccc', lw=0.7, alpha=0.8, zorder=0)
            if phi_deg != 0:
                ax.text(xs[-1] - 0.05, ys[-1], f'{phi_deg}°', fontsize=8, ha='right', va='center', color='#888')

    # equator (φ = 0) is a straight line y=0
    xs, ys = _mollweide_forward(np.linspace(-np.pi, np.pi, 361), np.zeros(361))
    ax.plot(xs, ys, color=equator_color, lw=1.2, alpha=0.9, label='equator (apical)')

    # prime meridian (λ = 0) is the central vertical curve x=0
    xs0, ys0 = _mollweide_forward(np.zeros(241), np.linspace(-np.pi/2, np.pi/2, 241))
    ax.plot(xs0, ys0, color=meridian_color, lw=1.2, alpha=0.9, label='0° meridian (a1)')

    # plot points
    if x_can.size:
        ax.plot(x_can, y_can, **can_style)
    if x_sin.size:
        ax.plot(x_sin, y_sin, **sin_style)

    # annotate poles (map to x=0, y=±√2)
    ax.plot([0], [ R_y], **pole_style)
    ax.plot([0], [-R_y], **pole_style)

    # cosmetics
    ax.set_xlim(-R_x * 1.02, R_x * 1.02)
    ax.set_ylim(-R_y * 1.02, R_y * 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', frameon=False)
    ax.grid(False)

    return fig, ax