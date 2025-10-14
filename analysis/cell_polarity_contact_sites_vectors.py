# %%
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from joblib import Parallel, delayed, parallel_backend
import numpy as np
import pandas as pd
from funlib.geometry import Roi, Coordinate
from tqdm import tqdm
import os
import time
import yaml
from nematic_voxel import calculate_and_extract_nematic_results, locally_average_nematic_vectors_from_dataframe
from simple_vector_measures import polarity_isotropic
import numpy as np
import json 
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

data_dir = cfg["FINAL_DATA_DIR"]
analysis_dir = cfg["ANALYSIS_DIR"]
skeleton_dir = cfg["SKELETON_DIR"]


def process_cell(
    cell_id: int,
    cell_df: pd.DataFrame,
    cell_idi: ImageDataInterface,
    canaliculi_contacts_idi: ImageDataInterface,
    sinusoid_contacts_idi: ImageDataInterface,
    return_points=False,
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
        
        # Append com to results
        result["COM Z (nm)"] = cell_com[0]
        result["COM Y (nm)"] = cell_com[1]
        result["COM X (nm)"] = cell_com[2]

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

        can_nematic_results = calculate_and_extract_nematic_results(
            can_pts,
            cell_com,
        )
        sin_nematic_results = calculate_and_extract_nematic_results(
            sin_pts,
            cell_com,
        )
        for n,r in zip(["Canaliculi", "Sinusoids"], [can_nematic_results, sin_nematic_results]):
            vec_name = "a" if n=="Canaliculi" else "b"
            for vec_num in [1,2]:
                result[f"{n} {vec_name}{vec_num} Z"] = r[f"vec{vec_num}"][0]
                result[f"{n} {vec_name}{vec_num} Y"] = r[f"vec{vec_num}"][1]
                result[f"{n} {vec_name}{vec_num} X"] = r[f"vec{vec_num}"][2]
                result[f"{n} {vec_name}{vec_num} sigma"] = r[f"sigma{vec_num}"]


    except Exception as e:
        print(f"[ERROR] cell {cell_id} raised: {e!r}")
        raise
    if return_points:
        # Append contact points to results as strings
       return cell_com, can_pts, sin_pts, result
    return result

# %%
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
        
        for base in ["Canaliculi a1", "Canaliculi a2", "Sinusoids b1", "Sinusoids b2"]:
            # Smooth nematic vectors and add to dataframe
            smoothed = locally_average_nematic_vectors_from_dataframe(
                polarity_df,
                base=base,
                sigma_column = f"{base} sigma",
                smoothing_std_um=20.0,
            )
            smoothed = smoothed['V']
            polarity_df[f"{base} Smoothed Z"] = smoothed[:, 0]
            polarity_df[f"{base} Smoothed Y"] = smoothed[:, 1]
            polarity_df[f"{base} Smoothed X"] = smoothed[:, 2]
            
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
# # Example usage - uncomment to run plotting
# for dataset in ["jrc_mus-liver-zon-1", "jrc_mus-liver-zon-2"]:
#     polarity_csv = f"{analysis_dir}/{dataset}/tmp_secondary_results/cell_polarity_vectors.csv"
#     cell_csv = f"{analysis_dir}/{dataset}/cell_assignments/cell.csv"
#     output_dir = f"{analysis_dir}/{dataset}/tmp_secondary_results/plots"
    
#     # Plot 2D vector fields - BINNED AND WEIGHTED (default)
#     # XZ plane (averaged along Y)
#     # plot_vector_field_2d(polarity_csv, cell_csv, grid_size_um=None, plane='xz',
#     #                      output_dir=output_dir, dataset_name=dataset, arrow_scale=.05,
#     #                      use_weighting=True)
    
#     # Optionally: Plot UNWEIGHTED versions for comparison
#     # plot_vector_field_2d(polarity_csv, cell_csv, grid_size_um=25, plane='xz',
#     #                      output_dir=output_dir, dataset_name=dataset, arrow_scale=.025,
#     #                      use_weighting=False)
    
#     # Optionally: Plot WITHOUT BINNING (per-cell vectors at COM)
#     # plot_vector_field_2d(polarity_csv, cell_csv, grid_size_um=None, plane='xz',
#     #                      output_dir=output_dir, dataset_name=dataset, arrow_scale=.025)
#     # plot_vector_field_2d(polarity_csv, cell_csv, grid_size_um=None, plane='yz',
#     #                      output_dir=output_dir, dataset_name=dataset, arrow_scale=.025)
    
#     # Original polarity vs Z plots
#     # plot_polarity_vs_z(polarity_csv, cell_csv, bin_size_um=25, average_axis='x', 
#     #                    output_dir=output_dir, dataset_name=dataset)
#     # plot_polarity_vs_z(polarity_csv, cell_csv, bin_size_um=25, average_axis='y', 
#     #                    output_dir=output_dir, dataset_name=dataset)

from vis import plot_projected_axes_from_csv
for dataset in ["jrc_mus-liver-zon-1", "jrc_mus-liver-zon-2"]:
    polarity_csv = f"{analysis_dir}/{dataset}/tmp_secondary_results/cell_polarity_vectors.csv"
    for base in ["Canaliculi a1", "Canaliculi a2","Sinusoids b1", "Sinusoids b2"]:
        for smoothing_std in [None, 20]:
            print(f"Plotting {base} for dataset {dataset} with smoothing={smoothing_std}")
            plot_projected_axes_from_csv(csv_path=polarity_csv, bases=[base], plane="YZ", length_3d=5000.0, smoothing_std_um=smoothing_std, colors_for_bases={base:"green"}, com_alpha=0.1)

# %% plot individual cells
from vis import plot_projected_points_with_two_vectors, plot_mollweide_polarity
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
cell_com, can_pts, sin_pts, results = process_cell(
    cell_id=1131,
    cell_df=cell_df,
    cell_idi=cell_idi,
    canaliculi_contacts_idi=canaliculi_contacts_idi,
    sinusoid_contacts_idi=sinusoid_contacts_idi,
    return_points=True,
)
a1 = np.array([results["Canaliculi a1 Z"], results["Canaliculi a1 Y"], results["Canaliculi a1 X"]])
a2 = np.array([results["Canaliculi a2 Z"], results["Canaliculi a2 Y"], results["Canaliculi a2 X"]])
b1 = np.array([results["Sinusoids b1 Z"], results["Sinusoids b1 Y"], results["Sinusoids b1 X"]])
b2 = np.array([results["Sinusoids b2 Z"], results["Sinusoids b2 Y"], results["Sinusoids b2 X"]])
plot_projected_points_with_two_vectors(can_pts, cell_com, vec1=a1, vec2=a2,sigma1=results["Canaliculi a1 sigma"], sigma2=results["Canaliculi a2 sigma"], vector_scale=3, point_size_proj=0.5)
# %%
# plot_projected_points_with_two_vectors(sin_pts, cell_com, vec1=b1, vec2=b2,sigma1=results["Sinusoids b1 sigma"], sigma2=results["Sinusoids b2 sigma"], vector_scale=3, point_size_proj=0.5)

# %%
plot_mollweide_polarity(a1, b1, cell_com, can_pts=can_pts)
# %%
from vis import plot_projected_axes_from_csv
for dataset in ["jrc_mus-liver-zon-1", "jrc_mus-liver-zon-2"]:
    for smoothing_std in [None, 20]:
        polarity_csv = f"{analysis_dir}/{dataset}/tmp_secondary_results/cell_polarity_vectors.csv"
        base = "Canaliculi a1"
        plot_projected_axes_from_csv(csv_path=polarity_csv, bases=[base], plane="YZ", length_3d=5000.0, smoothing_std_um=smoothing_std, colors_for_bases={base:"green"}, com_alpha=0.0, mesh_path="/nrs/cellmap/ackermand/meshes/multiresolution/jrc_mus-liver-zon-1/veins/mesh_lods/s5")

# %%
from vis import plot_projected_axes_from_csv
for dataset in ["jrc_mus-liver-zon-1", "jrc_mus-liver-zon-2"]:
    smoothing_std = None
    polarity_csv = f"{analysis_dir}/{dataset}/tmp_secondary_results/cell_polarity_vectors.csv"
    base = "Canaliculi a1 Smoothed"
    plot_projected_axes_from_csv(csv_path=polarity_csv, bases=[base], plane="YZ", length_3d=5000.0, smoothing_std_um=smoothing_std, colors_for_bases={base:"green"}, com_alpha=0.0, mesh_path="/nrs/cellmap/ackermand/meshes/multiresolution/jrc_mus-liver-zon-1/veins/mesh_lods/s5")

# %%
