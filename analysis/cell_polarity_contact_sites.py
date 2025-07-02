# %%
# import yaml
# with open("config.yaml", "r") as f:
#     cfg = yaml.safe_load(f)

# data_dir   = cfg["DATA_DIR"]
# analysis_dir = cfg["ANALYSIS_DIR"]
# skeleton_dir = cfg["SKELETON_DIR"]

# from cellmap_analyze.util.image_data_interface import ImageDataInterface
# from funlib.geometry import Coordinate, Roi
# import numpy as np
# from scipy.spatial import KDTree
# import pandas as pd
# from tqdm import tqdm

# canaliculi_contacts_idi = ImageDataInterface(
#     "{data_dir}/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2_contact-sites.zarr/canoliculi_cc_close_raw_mask_filled_filteredIDs_cell_contacts/s0"
# )
# sinusoid_contacts_idi = ImageDataInterface(
#     "{data_dir}/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2_contact-sites.zarr/ecs_masked_with_canaliculi_filteredIDS_and_cell_filled_cell_contacts/s0"
# )
# cell_idi = ImageDataInterface(
#     "{data_dir}/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr/cell/s0",
#     output_voxel_size=Coordinate(128, 128, 128),
# )
# cell_df = pd.read_csv(
#     f"{analysis_dir}/jrc_mus-liver-zon-2/cell.csv"
# )

# organelle_df = pd.read_csv(
#     f"{analysis_dir}/jrc_mus-liver-zon-2/cell_assignments/contact_sites/er_mito_contacts_assigned_to_containing_cell.csv"
# )
# # add column to organelle_df
# organelle_df["Nearest Canaliculi-Cell Contact Distance (nm)"] = np.nan
# organelle_df["Nearest Sinusoid-Cell Contact Distance (nm)"] = np.nan
# unique_cells = cell_df["Object ID"].to_numpy()

# for cell_id in [2]:  # tqdm(unique_cells):
#     mask = organelle_df["Cell ID"] == cell_id
#     organelle_coms = organelle_df[mask][
#         ["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]
#     ].to_numpy()

#     mins = cell_df[cell_df["Object ID"] == cell_id][
#         ["MIN Z (nm)", "MIN Y (nm)", "MIN X (nm)"]
#     ].to_numpy()[0]
#     maxs = cell_df[cell_df["Object ID"] == cell_id][
#         ["MAX Z (nm)", "MAX Y (nm)", "MAX X (nm)"]
#     ].to_numpy()[0]
#     mins -= cell_idi.voxel_size / 2
#     maxs -= (
#         cell_idi.voxel_size / 2
#     )  # here max is inclusive, so need to add voxel size back
#     roi = Roi(mins, (maxs - mins) + cell_idi.voxel_size)

#     cell = cell_idi.to_ndarray_ts(roi=roi) == cell_id
#     canaliculi_contacts = canaliculi_contacts_idi.to_ndarray_ts(roi=roi) > 0
#     sinusoid_contacts = sinusoid_contacts_idi.to_ndarray_ts(roi=roi) > 0

#     canaliculi_contacts = cell & canaliculi_contacts
#     sinusoid_contacts = cell & sinusoid_contacts

#     canaliculi_contacts = (
#         np.array(np.where(canaliculi_contacts)).T * cell_idi.output_voxel_size[0] + mins
#     )
#     sinusoid_contacts = (
#         np.array(np.where(sinusoid_contacts)).T * cell_idi.output_voxel_size[0] + mins
#     )
#     tree = KDTree(canaliculi_contacts)
#     canaliculi_dists, _ = tree.query(organelle_coms, k=1)
#     tree = KDTree(sinusoid_contacts)
#     sinusoid_dists, _ = tree.query(organelle_coms, k=1)
#     organelle_df.loc[mask, "Nearest Canaliculi-Cell Contact Distance (nm)"] = (
#         canaliculi_dists
#     )
#     organelle_df.loc[mask, "Nearest Sinusoid-Cell Contact Distance (nm)"] = (
#         sinusoid_dists
#     )
# Query the nearest neighbor in B for each point of A
# dist, idx = tree.query(A)
# 'dist' is the array of distances, 'idx' are the nearest-neighbor indices in B
# %%
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from joblib import Parallel, delayed
from scipy.spatial import KDTree
import numpy as np
import pandas as pd
from funlib.geometry import Roi, Coordinate
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import os

import yaml
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

data_dir   = cfg["DATA_DIR"]
analysis_dir = cfg["ANALYSIS_DIR"]
skeleton_dir = cfg["SKELETON_DIR"]

# dataset = "jrc_mus-liver-zon-2"
dataset = "jrc_mus-liver-zon-1"


def process_cell(
    sub_df: pd.githubDataFrame,
    cell_df: pd.DataFrame,
    cell_idi: ImageDataInterface,
    canaliculi_contacts_idi: ImageDataInterface,
    sinusoid_contacts_idi: ImageDataInterface,
) -> pd.DataFrame:
    cell_id = sub_df["Cell ID"].iat[0]

    # 1) get COM coords
    organelle_coms = sub_df[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].to_numpy()

    # 2) build ROI from cell_df
    mins = cell_df.loc[
        cell_df["Object ID"] == cell_id,
        ["MIN Z (nm)", "MIN Y (nm)", "MIN X (nm)"],
    ].to_numpy()[0]
    maxs = cell_df.loc[
        cell_df["Object ID"] == cell_id,
        ["MAX Z (nm)", "MAX Y (nm)", "MAX X (nm)"],
    ].to_numpy()[0]
    mins -= cell_idi.voxel_size / 2
    maxs -= cell_idi.voxel_size / 2
    roi = Roi(mins, (maxs - mins) + cell_idi.voxel_size)

    # 3) pull out masks & world‐coords
    cell_mask = cell_idi.to_ndarray_ts(roi=roi) == cell_id
    can_mask = cell_mask & (canaliculi_contacts_idi.to_ndarray_ts(roi=roi) > 0)
    sin_mask = cell_mask & (sinusoid_contacts_idi.to_ndarray_ts(roi=roi) > 0)

    can_pts = (np.argwhere(can_mask) * cell_idi.output_voxel_size[0]) + mins
    sin_pts = (np.argwhere(sin_mask) * cell_idi.output_voxel_size[0]) + mins

    # 4) nearest‐neighbor distances
    if organelle_coms.size == 0 or can_pts.size == 0:
        can_d = np.full(len(organelle_coms), np.inf)
    else:
        can_d, _ = (
            NearestNeighbors(n_neighbors=1).fit(can_pts).kneighbors(organelle_coms)
        )
    if organelle_coms.size == 0 or sin_pts.size == 0:
        sin_d = np.full(len(organelle_coms), np.inf)
    else:
        sin_d, _ = (
            NearestNeighbors(n_neighbors=1).fit(sin_pts).kneighbors(organelle_coms)
        )

    # 5) write back
    sub_df["Nearest Canaliculi-Cell Contact Distance (nm)"] = can_d.ravel()
    sub_df["Nearest Sinusoid-Cell Contact Distance (nm)"] = sin_d.ravel()
    return sub_df


if dataset == "jrc_mus-liver-zon-1":
    canaliculi_name = "canaliculi_filteredIDs_cc_cell_ecs_raw_endothelial_masks_filled"
    sinusoid_name = "ecs_masked_with_canaliculi_filteredIDs_cc_cell_ecs_raw_endothelial_masks_filled_filled"
    cell_name = "original_cell"
elif dataset == "jrc_mus-liver-zon-2":
    canaliculi_name = "canoliculi_cc_close_raw_mask_filled_filteredIDs"
    sinusoid_name = "ecs_masked_with_canaliculi_filteredIDS_and_cell_filled"
if __name__ == "__main__":
    canaliculi_contacts_idi = ImageDataInterface(
        f"{data_dir}/{dataset}/{dataset}_contact-sites.zarr/{canaliculi_name}_cell_contacts/s0"
    )
    sinusoid_contacts_idi = ImageDataInterface(
        f"{data_dir}/{dataset}/{dataset}_contact-sites.zarr/{sinusoid_name}_cell_contacts/s0"
    )

    cell_idi = ImageDataInterface(
        f"{data_dir}/{dataset}/{dataset}.zarr/{cell_name}/s0",
        output_voxel_size=Coordinate(128, 128, 128),
    )

    cell_df = pd.read_csv(
        f"{analysis_dir}/{dataset}/20250430_duplicate_ids/cell.csv"
    )
    unique_cells = cell_df["Object ID"].to_numpy()

    organelles = [
        "mito_with_skeleton",
        "perox",
        "ld",
        "nuc",
        "er",
        # "er_mito_contacts",
        # "er_perox_contacts",
        # "er_ld_contacts",
        # "ld_perox_contacts",
        # "mito_perox_contacts",
        # "mito_ld_contacts",
    ]
    for organelle in organelles:
        extra_dir = ""
        if organelle.endswith("_contacts"):
            extra_dir = "contact_sites/"

        organelle_df = pd.read_csv(
            f"{analysis_dir}/{dataset}/20250430_duplicate_ids/cell_assignments/{extra_dir}{organelle}_assigned_to_containing_cell.csv"
        )
        print("Processing organelle:", organelle)

        # add column to organelle_df
        organelle_df["Nearest Canaliculi-Cell Contact Distance (nm)"] = np.nan
        organelle_df["Nearest Sinusoid-Cell Contact Distance (nm)"] = np.nan

        # build one sub-DF per cell, skipping cell 0
        sub_dfs = [
            grp.copy()
            for cell_id, grp in organelle_df.groupby("Cell ID", sort=False)
            if cell_id != 0
        ]
        print("got sub-dfs")
        del organelle_df

        # run in parallel

        results = Parallel(n_jobs=-1, timeout=10000)(
            delayed(process_cell)(
                df,
                cell_df,
                cell_idi,
                canaliculi_contacts_idi,
                sinusoid_contacts_idi,
            )
            for df in tqdm(sub_dfs, desc=f"Cells processed for {organelle}")
        )

        # concatenate in original order and save
        # each piece has all original cols + distances  # restore the original row order
        organelle_df = pd.concat(results).sort_index()
        os.makedirs(
            f"{analysis_dir}/{dataset}/20250430_duplicate_ids/cell_assignments/{extra_dir}polarity/",
            exist_ok=True,
        )
        organelle_df.to_csv(
            f"{analysis_dir}/{dataset}/20250430_duplicate_ids/cell_assignments/{extra_dir}polarity/{organelle}_assigned_to_containing_cell_polarity.csv",
            index=False,
        )

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

organelles = [
    "mito_with_skeleton",
    "perox",
    "ld",
    "nuc",
    "er",
    # "er_mito_contacts",
    # "er_perox_contacts",
    # "er_ld_contacts",
    # "ld_perox_contacts",
    # "mito_perox_contacts",
    # "mito_ld_contacts",
]

# properties & error types
properties = ["radius mean (nm)", "lsp (nm)", "Volume (nm^3)", "Sphericity"]
errors = [("std", "±1 Std Dev"), ("ci", "95% CI")]

# total number of rows = 4 props * 2 error types + 1 count plot + 1 ratio plot
n_rows = len(properties) * len(errors) + 1 + 1
n_cols = len(organelles)
fig, axes = plt.subplots(
    n_rows, len(organelles), figsize=(6 * n_cols, 4 * n_rows), sharex=True
)

dataset = "jrc_mus-liver-zon-2"
for col, organelle in enumerate(organelles):
    extra_dir = ""
    if organelle.endswith("_contacts"):
        extra_dir = "contact_sites/"

    # --- load & prepare your DataFrame as before ---
    # if dataset == "jrc_mus-liver-zon-1":
    #     csv_path = f"{analysis_dir}/jrc_mus-liver-zon-1/20250430_duplicate_ids/{extra_dir}{organelle}.csv"
    # else:
    cell_df = pd.read_csv(
        f"{analysis_dir}/{dataset}/20250430_duplicate_ids/cell.csv"
    )
    organelle_df = pd.read_csv(
        f"{analysis_dir}/{dataset}/20250430_duplicate_ids/cell_assignments/{extra_dir}polarity/{organelle}_assigned_to_containing_cell_polarity.csv"
    )
    large_cells = cell_df.loc[
        cell_df["Volume (nm^3)"] >= 4e11,  # threshold
        "Object ID",  # the column holding the cell IDs
    ]
    organelle_df = organelle_df.loc[organelle_df["Cell ID"].isin(large_cells), :]

    organelle_df["Sphericity"] = (
        (np.pi ** (1 / 3))
        * (6 * organelle_df["Volume (nm^3)"]) ** (2 / 3)
        / organelle_df["Surface Area (nm^2)"]
    )
    org_df = organelle_df.copy()
    # if dataset == "jrc_mus-liver-zon-1":
    #     org_df["category"] = "Canaliculi"
    #     # set first one to "Sinusoid"
    #     org_df.loc[0, "category"] = "Sinusoid"
    # else:
    can_col = "Nearest Canaliculi-Cell Contact Distance (nm)"
    sin_col = "Nearest Sinusoid-Cell Contact Distance (nm)"
    finite_mask = np.isfinite(org_df[can_col]) & np.isfinite(org_df[sin_col])
    org_df["Category"] = "None"
    org_df.loc[finite_mask, "Category"] = np.where(
        org_df.loc[finite_mask, can_col] < org_df.loc[finite_mask, sin_col],
        "Canaliculi",
        "Sinusoid",
    )
    org_df_for_writing = org_df.copy()
    # remove category None
    org_df_for_writing = org_df_for_writing[org_df["Category"] != "None"]
    org_df_for_writing.to_csv(
        f"{analysis_dir}/{dataset}/20250430_duplicate_ids/forDaniel/{organelle}.csv",
        index=False,
    )

    # bins & midpoints (shared)
    num_bins = 20
    z_all = org_df["COM Z (nm)"]
    bins = np.linspace(z_all.min(), z_all.max(), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    org_df["z_bin"] = pd.cut(
        z_all,
        bins=bins,
        labels=bin_centers,
        include_lowest=True,  # <<— now [bins[0], bins[1]] for the first bin
    )

    # helper to compute per-bin stats
    def compute_stats(df, prop, error="std", ci_level=0.95):
        # aggregate
        stats = (
            df.groupby(["Category", "z_bin"], observed=True)
            .agg(
                mean_val=(prop, "mean"),
                std_val=(prop, "std"),
                n=("COM Z (nm)", "count"),
            )
            .reset_index()
        )
        stats["z_center"] = stats["z_bin"].astype(float)
        # error band
        if error == "std":
            stats["err"] = stats["std_val"]
        elif error == "ci":
            z_score = norm.ppf(1 - (1 - ci_level) / 2)
            stats["err"] = z_score * stats["std_val"] / np.sqrt(stats["n"])
        else:
            raise ValueError
        return stats

    row = 0
    for err_code, err_label in errors:
        for prop in properties:
            if prop in org_df.columns:
                ax = axes[row, col]
                stats = compute_stats(org_df, prop, error=err_code, ci_level=0.95)
                for cat in stats["Category"].unique():
                    if cat == "None":
                        continue
                    sub = stats[stats["Category"] == cat]
                    ax.plot(sub["z_center"], sub["mean_val"], label=f"{cat} mean")
                    ax.fill_between(
                        sub["z_center"],
                        sub["mean_val"] - sub["err"],
                        sub["mean_val"] + sub["err"],
                        alpha=0.3,
                    )
                ax.set_ylabel(prop)
                ax.set_title(f"{organelle} {prop}")
                if row == 0:
                    ax.legend(loc="upper right")
            row += 1

    # final row: counts + difference
    # compute counts per bin
    count_df = (
        org_df.groupby(["Category", "z_bin"], observed=True)
        .size()
        .reset_index(name="count")
    )
    count_df["z_center"] = count_df["z_bin"].astype(float)

    # pivot so that each z_center is a row and each category is a column
    pivot_df = (
        count_df.pivot(index="z_center", columns="Category", values="count")
        .fillna(0)  # missing (cat, bin) → 0
        .sort_index()  # ensure bins are in order
    )
    # compute the difference
    pivot_df["diff"] = pivot_df["Canaliculi"] - pivot_df["Sinusoid"]
    pivot_df["ratio"] = pivot_df["Canaliculi"] / pivot_df["Sinusoid"]

    ax = axes[row, col]
    for cat in ["Canaliculi", "Sinusoid"]:
        sub = count_df[count_df["Category"] == cat]
        ax.plot(
            pivot_df.index,
            pivot_df[cat],
            marker="o",
            label=f"{cat} (n={int(pivot_df[cat].sum())})",
        )
    # difference
    can = count_df[count_df["Category"] == "Canaliculi"]["count"].to_numpy()
    sin = count_df[count_df["Category"] == "Sinusoid"]["count"].to_numpy()
    ax.plot(
        pivot_df.index,
        pivot_df["diff"],
        marker="o",
        label=f"Canaliculi–Sinusoid (n={can.sum() + sin.sum()})",
    )
    ax.set_xlabel("COM Z (nm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Number of {organelle}")
    ax.legend(loc="upper right")
    row += 1

    # ratio
    ax = axes[row, col]
    ax.plot(
        pivot_df.index,
        pivot_df["ratio"],
        marker="o",
        label=f"{organelle} count ratio: Canaliculi/Sinusoid (n={can.sum() + sin.sum()})",
    )
    ax.set_xlabel("COM Z (nm)")
    ax.set_ylabel("Canaliculi/Sinusoid Ratio")
    ax.set_title("Ratio vs. Z-bin")
plt.tight_layout()
plt.show()
# %%
# import matplotlib.pyplot as plt
# from scipy.stats import binned_statistic_2d


# z = organelle_df["COM Z (nm)"]
# canaliculi = organelle_df["Nearest Canaliculi-Cell Contact Distance (nm)"]
# sinusoid = organelle_df["Nearest Sinusoid-Cell Contact Distance (nm)"]

# # mask = True only where both x and y are finite
# mask = np.isfinite(y)


# # z = (
# #     (np.pi ** (1 / 3))
# #     * (6 * df["Volume (nm^3)"]) ** (2 / 3)
# #     / df["Surface Area (nm^2)"]
# # )  # The data we want to average in each bin
# z = organelle_df["radius mean (nm)"]
# # z = df["Volume (nm^3)"] / df["lsp (nm)"]
# # Compute the 2D binned statistic (mean in this example)
# # apply the mask
# x = x[mask]
# y = y[mask]
# z = z[mask]
# statistic, xedges, yedges, binnumber = binned_statistic_2d(
#     x,
#     y,
#     z,
#     statistic="median",  # could be 'mean', 'sum', 'median', 'count', etc.
#     bins=75,
# )
# valid_vals = statistic[np.isfinite(statistic)]
# vmin_5, vmax_95 = np.percentile(valid_vals, [5, 95])


# # Plot using pcolormesh
# plt.figure(figsize=(8, 6))

# # Note that statistic has shape (n_bins_x, n_bins_y),
# # but pcolormesh expects the transpose for normal orientation
# mesh = plt.pcolormesh(
#     xedges,
#     yedges,
#     statistic.T,
#     cmap="viridis",
#     vmin=vmin_5,
#     vmax=vmax_95,  # or you can do .transpose()
# )
# tick_ratios = [10**i for i in range(-3, 4)]

# # 2) Convert them to log-space for their actual positions
# tick_positions = np.log10(tick_ratios)

# # 3) Apply them to the y-axis (since log_ratio is on y)
# plt.yticks(tick_positions, tick_ratios)
# cb = plt.colorbar(mesh)
# cb.set_label("radius (nm)")

# plt.xlabel("COM Z (nm)")
# plt.ylabel("canaliculi distance / sinusoid distance")
# plt.title("2D Histogram of Median radius in each (Z, ratio) bin")
# plt.axhline(y=0, color="red", linestyle="--", label="ratio = 1")
# # plt.ylim([np.log10(1/2),np.log10(2)])
# plt.tight_layout()
# plt.show()

# # %%
# %%
import pandas as pd
pd.read_csv("")