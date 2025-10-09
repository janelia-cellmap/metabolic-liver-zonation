
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd
import os

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

