# %%
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from joblib import Parallel, delayed, parallel_backend
import numpy as np
import pandas as pd
from funlib.geometry import Roi, Coordinate
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import os
import time
import yaml

import numpy as np

def unit_vectors(points, center):
    """
    Convert contact voxel coordinates into unit vectors
    pointing from the cell center toward each contact voxel.
    """
    v = points - center[None, :]
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def mean_dir_R(U):
    """
    Compute mean direction (unit vector) and concentration (R)
    for a set of 3D unit vectors.
    """
    if len(U) == 0:
        return np.array([np.nan, np.nan, np.nan]), 0.0
    s = U.sum(axis=0)
    s_norm = np.linalg.norm(s)
    R = s_norm / len(U)
    m = s / s_norm if s_norm > 0 else np.array([np.nan, np.nan, np.nan])
    return m, R


def cone_percentiles(U, axis):
    """
    Compute 50%, 68%, 95% angular half-cone spread (degrees)
    around a given mean axis.
    """
    if len(U) == 0 or not np.isfinite(axis).all():
        return np.array([np.nan, np.nan, np.nan])
    theta = np.arccos(np.clip(U @ axis, -1.0, 1.0))
    q = np.percentile(theta, [50, 68, 95])
    return np.rad2deg(q)


def polarity_isotropic(canal_vox, sinus_vox, com):
    """
    Compute 3D polarity based on canaliculi and sinusoid contact voxels.

    Handles all cases:
      - both present
      - only one present
      - neither present
    """
    # Convert inputs to arrays
    canal_vox = np.asarray(canal_vox)
    sinus_vox = np.asarray(sinus_vox)
    com = np.asarray(com)

    # Step 1: unit vectors from COM to contact voxels
    Uc = unit_vectors(canal_vox, com) if canal_vox.size else np.empty((0, 3))
    Us = unit_vectors(sinus_vox, com) if sinus_vox.size else np.empty((0, 3))

    # Step 2: mean direction + concentration for each
    mc, Rc = mean_dir_R(Uc)
    ms, Rs = mean_dir_R(Us)

    # Step 3: compute polarity depending on which contacts exist
    if Rc > 0 and Rs > 0:
        # Both canaliculi and sinusoid contacts present
        p_raw = Rc * mc - Rs * ms
    elif Rc > 0 and Rs == 0:
        # Only canaliculi contacts → point toward canaliculi
        p_raw = Rc * mc
    elif Rs > 0 and Rc == 0:
        # Only sinusoid contacts → point away from sinusoids
        p_raw = -Rs * ms
    else:
        # Neither contact type present
        p_raw = np.array([np.nan, np.nan, np.nan])

    # Step 4: compute polarity strength and normalize axis
    strength = np.linalg.norm(p_raw) if np.all(np.isfinite(p_raw)) else 0.0
    polarity_axis = (
        p_raw / strength if strength > 0 else np.array([np.nan, np.nan, np.nan])
    )

    # Step 5: angular spreads
    coneC = cone_percentiles(Uc, mc)
    coneS = cone_percentiles(Us, ms)

    # Step 6: pack everything into a clean dictionary
    return {
        "canaliculi": {
            "mean_dir": mc,
            "R": Rc,
            "cone_deg": {
                "median": coneC[0],
                "p68": coneC[1],
                "p95": coneC[2],
            },
        },
        "sinusoids": {
            "mean_dir": ms,
            "R": Rs,
            "cone_deg": {
                "median": coneS[0],
                "p68": coneS[1],
                "p95": coneS[2],
            },
        },
        "polarity_axis": polarity_axis,   # final polarity direction (unit vector)
        "polarity_strength": strength     # scalar (0–2 typical)
    }

import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd

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


# def plot_polarity_vs_z(
#     polarity_csv_path: str,
#     cell_csv_path: str,
#     bin_size_um: float = 25.0,
#     average_axis: str = 'x',  # 'x' or 'y'
#     output_dir: str = None,
#     dataset_name: str = None
# ):
#     """
#     Plot polarity directions averaged along x or y axis, binned in z.
    
#     Parameters:
#     -----------
#     polarity_csv_path : str
#         Path to the polarity vectors CSV file
#     cell_csv_path : str
#         Path to the cell CSV file with COM coordinates
#     bin_size_um : float
#         Bin size in micrometers (default 25 μm)
#     average_axis : str
#         Axis to average along ('x' or 'y')
#     output_dir : str
#         Directory to save plots (optional)
#     dataset_name : str
#         Name of dataset for plot titles
#     """
#     # Load data
#     polarity_df = pd.read_csv(polarity_csv_path)
#     cell_df = pd.read_csv(cell_csv_path)
    
#     # Merge to get COM coordinates
#     merged_df = polarity_df.merge(
#         cell_df[['Object ID', 'COM Z (nm)', 'COM Y (nm)', 'COM X (nm)']],
#         left_on='Cell ID',
#         right_on='Object ID',
#         how='left'
#     )
    
#     # Filter out cells without valid polarity
#     valid_df = merged_df[merged_df['Polarity Strength'].notna()].copy()
    
#     # Convert bin size from μm to nm
#     bin_size_nm = bin_size_um * 1000
    
#     # Create bins based on Z coordinate
#     z_min = valid_df['COM Z (nm)'].min()
#     z_max = valid_df['COM Z (nm)'].max()
#     z_bins = np.arange(z_min, z_max + bin_size_nm, bin_size_nm)
#     z_bin_centers = (z_bins[:-1] + z_bins[1:]) / 2 / 1000  # Convert to μm for plotting
    
#     # Assign bins
#     valid_df['z_bin'] = pd.cut(valid_df['COM Z (nm)'], bins=z_bins, labels=False)
    
#     # Prepare results storage
#     results = {
#         'z_centers': [],
#         'polarity_mean_z': [],
#         'polarity_mean_lateral': [],  # x or y depending on average_axis
#         'polarity_strength_mean': [],
#         'polarity_strength_std': [],
#         'n_cells': [],
#         'canaliculi_R_mean': [],
#         'sinusoids_R_mean': []
#     }
    
#     # Calculate averages for each bin
#     for bin_idx in range(len(z_bins) - 1):
#         bin_data = valid_df[valid_df['z_bin'] == bin_idx]
        
#         if len(bin_data) == 0:
#             continue
            
#         results['z_centers'].append(z_bin_centers[bin_idx])
#         results['n_cells'].append(len(bin_data))
        
#         # Average polarity vectors
#         polarity_z = bin_data['Polarity Axis Z'].mean()
#         if average_axis.lower() == 'x':
#             polarity_lateral = bin_data['Polarity Axis Y'].mean()
#         else:  # 'y'
#             polarity_lateral = bin_data['Polarity Axis X'].mean()
        
#         results['polarity_mean_z'].append(polarity_z)
#         results['polarity_mean_lateral'].append(polarity_lateral)
#         results['polarity_strength_mean'].append(bin_data['Polarity Strength'].mean())
#         results['polarity_strength_std'].append(bin_data['Polarity Strength'].std())
#         results['canaliculi_R_mean'].append(bin_data['Canaliculi R'].mean())
#         results['sinusoids_R_mean'].append(bin_data['Sinusoids R'].mean())
    
#     # Create figure with subplots
#     fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
#     lateral_label = 'Y' if average_axis.lower() == 'x' else 'X'
#     title_suffix = f" (averaged along {average_axis.upper()})"
    
#     # Plot 1: Polarity vector components vs Z
#     ax = axes[0, 0]
#     ax.plot(results['z_centers'], results['polarity_mean_z'], 'o-', label='Polarity Z component', color='blue')
#     ax.plot(results['z_centers'], results['polarity_mean_lateral'], 's-', label=f'Polarity {lateral_label} component', color='red')
#     ax.axhline(0, color='k', linestyle='--', alpha=0.3)
#     ax.set_xlabel('Z position (μm)', fontsize=12)
#     ax.set_ylabel('Mean polarity component', fontsize=12)
#     ax.set_title(f'Polarity Vector Components vs Z{title_suffix}', fontsize=14)
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     # Plot 2: Polarity strength vs Z
#     ax = axes[0, 1]
#     ax.errorbar(results['z_centers'], results['polarity_strength_mean'], 
#                 yerr=results['polarity_strength_std'], fmt='o-', capsize=5, color='purple')
#     ax.set_xlabel('Z position (μm)', fontsize=12)
#     ax.set_ylabel('Mean polarity strength', fontsize=12)
#     ax.set_title(f'Polarity Strength vs Z{title_suffix}', fontsize=14)
#     ax.grid(True, alpha=0.3)
    
#     # Plot 3: Coherence measures (R values) vs Z
#     ax = axes[1, 0]
#     ax.plot(results['z_centers'], results['canaliculi_R_mean'], 'o-', label='Canaliculi R', color='green')
#     ax.plot(results['z_centers'], results['sinusoids_R_mean'], 's-', label='Sinusoids R', color='orange')
#     ax.set_xlabel('Z position (μm)', fontsize=12)
#     ax.set_ylabel('Mean R (coherence)', fontsize=12)
#     ax.set_title(f'Contact Coherence vs Z{title_suffix}', fontsize=14)
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     # Plot 4: Number of cells per bin
#     ax = axes[1, 1]
#     ax.bar(results['z_centers'], results['n_cells'], width=bin_size_um*0.8, alpha=0.6, color='gray')
#     ax.set_xlabel('Z position (μm)', fontsize=12)
#     ax.set_ylabel('Number of cells', fontsize=12)
#     ax.set_title(f'Cell Count per {bin_size_um}μm Bin', fontsize=14)
#     ax.grid(True, alpha=0.3)
    
#     plt.suptitle(f'Cell Polarity Analysis - {dataset_name}' if dataset_name else 'Cell Polarity Analysis', 
#                  fontsize=16, y=1.00)
#     plt.tight_layout()
    
#     # Save if output directory provided
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         output_file = os.path.join(output_dir, f'polarity_vs_z_avg_{average_axis}_{dataset_name}.png')
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#         print(f"Saved plot to: {output_file}")
    
#     plt.show()
    
#     return results
# end chatgpt
# %%
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

data_dir = cfg["FINAL_DATA_DIR"]
analysis_dir = cfg["ANALYSIS_DIR"]
skeleton_dir = cfg["SKELETON_DIR"]

# %%
def process_cell(
    cell_id: int,
    cell_df: pd.DataFrame,
    cell_idi: ImageDataInterface,
    canaliculi_contacts_idi: ImageDataInterface,
    sinusoid_contacts_idi: ImageDataInterface,
) -> dict:
    try:
        # 1) Get cell COM coords
        cell_com = cell_df.loc[
            cell_df["Object ID"] == cell_id,
            ["COM Z (nm)", "COM Y (nm)", "COM X (nm)"],
        ].to_numpy()[0]

        # 2) build ROI from cell_df
        mins = cell_df.loc[
            cell_df["Object ID"] == cell_id,
            ["MIN Z (nm)", "MIN Y (nm)", "MIN X (nm)"],
        ].to_numpy()[0]
        maxs = cell_df.loc[
            cell_df["Object ID"] == cell_id,
            ["MAX Z (nm)", "MAX Y (nm)", "MAX X (nm)"],
        ].to_numpy()[0]
        # ensure cell is contained within region
        # get with reference to top left corner, so subtract half box size
        center_on_voxel = cell_idi.voxel_size / 2
        mins -= center_on_voxel
        maxs -= center_on_voxel
        # extra padding
        mins -= cell_idi.voxel_size
        maxs += cell_idi.voxel_size
        roi = Roi(mins, (maxs - mins) + cell_idi.voxel_size[0])

        # 3) pull out masks & world‐coords
        recenter_on_128nm_voxel = center_on_voxel / 2
        cell_mask = cell_idi.to_ndarray_ts(roi=roi) == cell_id
        can_mask = cell_mask & (canaliculi_contacts_idi.to_ndarray_ts(roi=roi) > 0)
        sin_mask = cell_mask & (sinusoid_contacts_idi.to_ndarray_ts(roi=roi) > 0)

        can_pts = (
            (np.argwhere(can_mask) * cell_idi.output_voxel_size[0])
            + mins
            + recenter_on_128nm_voxel
        )
        sin_pts = (
            (np.argwhere(sin_mask) * cell_idi.output_voxel_size[0])
            + mins
            + recenter_on_128nm_voxel
        )

        # 4) Calculate polarity if we have both contact types
        result = {"Cell ID": cell_id}
        
        #if can_pts.size > 0 and sin_pts.size > 0:
        polarity_data = polarity_isotropic(can_pts, sin_pts, cell_com)
        
        # Canaliculi data
        result["Canaliculi Mean Dir Z"] = polarity_data["canaliculi"]["mean_dir"][0]
        result["Canaliculi Mean Dir Y"] = polarity_data["canaliculi"]["mean_dir"][1]
        result["Canaliculi Mean Dir X"] = polarity_data["canaliculi"]["mean_dir"][2]
        result["Canaliculi R"] = polarity_data["canaliculi"]["R"]
        result["Canaliculi Cone Median Deg"] = polarity_data["canaliculi"]["cone_deg"]["median"]
        result["Canaliculi Cone P68 Deg"] = polarity_data["canaliculi"]["cone_deg"]["p68"]
        result["Canaliculi Cone P95 Deg"] = polarity_data["canaliculi"]["cone_deg"]["p95"]
        
        # Sinusoids data
        result["Sinusoids Mean Dir Z"] = polarity_data["sinusoids"]["mean_dir"][0]
        result["Sinusoids Mean Dir Y"] = polarity_data["sinusoids"]["mean_dir"][1]
        result["Sinusoids Mean Dir X"] = polarity_data["sinusoids"]["mean_dir"][2]
        result["Sinusoids R"] = polarity_data["sinusoids"]["R"]
        result["Sinusoids Cone Median Deg"] = polarity_data["sinusoids"]["cone_deg"]["median"]
        result["Sinusoids Cone P68 Deg"] = polarity_data["sinusoids"]["cone_deg"]["p68"]
        result["Sinusoids Cone P95 Deg"] = polarity_data["sinusoids"]["cone_deg"]["p95"]
        
        # Polarity axis and strength
        result["Polarity Axis Z"] = polarity_data["polarity_axis"][0]
        result["Polarity Axis Y"] = polarity_data["polarity_axis"][1]
        result["Polarity Axis X"] = polarity_data["polarity_axis"][2]
        result["Polarity Strength"] = polarity_data["polarity_strength"]
        
        result["Num Canaliculi Contact Voxels"] = len(can_pts)
        result["Num Sinusoid Contact Voxels"] = len(sin_pts)
        # else:
        #     # Set NaN values if we don't have both contact types
        #     for key in ["Canaliculi Mean Dir Z", "Canaliculi Mean Dir Y", "Canaliculi Mean Dir X",
        #                "Canaliculi R", "Canaliculi Cone Median Deg", "Canaliculi Cone P68 Deg", "Canaliculi Cone P95 Deg",
        #                "Sinusoids Mean Dir Z", "Sinusoids Mean Dir Y", "Sinusoids Mean Dir X",
        #                "Sinusoids R", "Sinusoids Cone Median Deg", "Sinusoids Cone P68 Deg", "Sinusoids Cone P95 Deg",
        #                "Polarity Axis Z", "Polarity Axis Y", "Polarity Axis X", "Polarity Strength"]:
        #         result[key] = np.nan
        #     result["Num Canaliculi Contact Voxels"] = len(can_pts) if can_pts.size > 0 else 0
        #     result["Num Sinusoid Contact Voxels"] = len(sin_pts) if sin_pts.size > 0 else 0
            
    except Exception as e:
        print(f"[ERROR] cell {cell_id} raised: {e!r}")
        raise
    return result


if __name__ == "__main__":
    for dataset in ["jrc_mus-liver-zon-1", "jrc_mus-liver-zon-2"]:
        print(f"\nProcessing dataset: {dataset}")
        
        canaliculi_contacts_idi = ImageDataInterface(
            f"{data_dir}/{dataset}/{dataset}.zarr//recon-1/labels/inference/segmentations/canaliculi_cell_contacts/s0"
        )
        sinusoid_contacts_idi = ImageDataInterface(
            f"{data_dir}/{dataset}/{dataset}.zarr//recon-1/labels/inference/segmentations/sinusoid_cell_contacts/s0"
        )

        cell_idi = ImageDataInterface(
            f"{data_dir}/{dataset}/{dataset}.zarr//recon-1/labels/inference/segmentations/cell/s0",
            output_voxel_size=Coordinate(128, 128, 128),
        )

        cell_df = pd.read_csv(f"{analysis_dir}/{dataset}/cell_assignments/cell.csv")
        unique_cells = cell_df["Object ID"].to_numpy()
        t0 = time.time()

        # run in parallel
        with parallel_backend("multiprocessing", n_jobs=-1):
            results = Parallel()(
                delayed(process_cell)(
                    cell_id,
                    cell_df,
                    cell_idi,
                    canaliculi_contacts_idi,
                    sinusoid_contacts_idi,
                )
                for cell_id in tqdm(unique_cells, desc=f"Processing cells")
            )

        # Convert results to DataFrame
        polarity_df = pd.DataFrame(results)
        
        # Save to CSV
        os.makedirs(
            f"{analysis_dir}/{dataset}/tmp_secondary_results",
            exist_ok=True,
        )
        output_path = f"{analysis_dir}/{dataset}/tmp_secondary_results/cell_polarity_vectors.csv"
        polarity_df.to_csv(output_path, index=False)
        
        print(f"Processed dataset {dataset} in {time.time() - t0:.2f} seconds.")
        print(f"Saved results to: {output_path}")
        print(f"Total cells processed: {len(polarity_df)}")
        print(f"Cells with valid polarity: {polarity_df['Polarity Strength'].notna().sum()}")



# %%
# Example usage - uncomment to run plotting
for dataset in ["jrc_mus-liver-zon-1", "jrc_mus-liver-zon-2"]:
    polarity_csv = f"{analysis_dir}/{dataset}/tmp_secondary_results/cell_polarity_vectors.csv"
    cell_csv = f"{analysis_dir}/{dataset}/cell_assignments/cell.csv"
    output_dir = f"{analysis_dir}/{dataset}/tmp_secondary_results/plots"
    
    # Plot 2D vector fields - BINNED AND WEIGHTED (default)
    # XZ plane (averaged along Y)
    plot_vector_field_2d(polarity_csv, cell_csv, grid_size_um=None, plane='xz',
                         output_dir=output_dir, dataset_name=dataset, arrow_scale=.05,
                         use_weighting=True)
    
    # Optionally: Plot UNWEIGHTED versions for comparison
    # plot_vector_field_2d(polarity_csv, cell_csv, grid_size_um=25, plane='xz',
    #                      output_dir=output_dir, dataset_name=dataset, arrow_scale=.025,
    #                      use_weighting=False)
    
    # Optionally: Plot WITHOUT BINNING (per-cell vectors at COM)
    # plot_vector_field_2d(polarity_csv, cell_csv, grid_size_um=None, plane='xz',
    #                      output_dir=output_dir, dataset_name=dataset, arrow_scale=.025)
    # plot_vector_field_2d(polarity_csv, cell_csv, grid_size_um=None, plane='yz',
    #                      output_dir=output_dir, dataset_name=dataset, arrow_scale=.025)
    
    # Original polarity vs Z plots
    # plot_polarity_vs_z(polarity_csv, cell_csv, bin_size_um=25, average_axis='x', 
    #                    output_dir=output_dir, dataset_name=dataset)
    # plot_polarity_vs_z(polarity_csv, cell_csv, bin_size_um=25, average_axis='y', 
    #                    output_dir=output_dir, dataset_name=dataset)


# %%
