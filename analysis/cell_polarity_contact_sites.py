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

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

data_dir = cfg["DATA_DIR"]
analysis_dir = cfg["ANALYSIS_DIR"]
skeleton_dir = cfg["SKELETON_DIR"]


def process_cell(
    sub_df: pd.DataFrame,
    cell_df: pd.DataFrame,
    cell_idi: ImageDataInterface,
    canaliculi_contacts_idi: ImageDataInterface,
    sinusoid_contacts_idi: ImageDataInterface,
) -> pd.DataFrame:
    try:
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
    except Exception as e:
        print(f"[ERROR] cell {cell_id} raised: {e!r}")
        raise
    return sub_df


if __name__ == "__main__":
    for dataset in ["jrc_mus-liver-zon-1", "jrc_mus-liver-zon-2"]:
        canaliculi_contacts_idi = ImageDataInterface(
            f"{data_dir}/{dataset}/{dataset}_contact-sites.zarr/canaliculi_cell_contacts/s0"
        )
        sinusoid_contacts_idi = ImageDataInterface(
            f"{data_dir}/{dataset}/{dataset}_contact-sites.zarr/sinusoid_cell_contacts/s0"
        )

        cell_idi = ImageDataInterface(
            f"{data_dir}/{dataset}/{dataset}.zarr/cell/s0",
            output_voxel_size=Coordinate(128, 128, 128),
        )

        cell_df = pd.read_csv(f"{analysis_dir}/{dataset}/cell_assignments/cell.csv")
        unique_cells = cell_df["Object ID"].to_numpy()

        organelles = [
            "er",
            "mito_with_skeleton",
            "perox",
            "ld",
            "nuc",
            "er_mito_contacts",
            "er_perox_contacts",
            "er_ld_contacts",
            "ld_perox_contacts",
            "mito_perox_contacts",
            "mito_ld_contacts",
        ]
        for organelle in organelles:
            extra_dir = ""
            if organelle.endswith("_contacts"):
                extra_dir = "contact_sites/"

            print(f"Processing dataset {dataset}, organelle: {organelle}...")
            organelle_df = pd.read_csv(
                f"{analysis_dir}/{dataset}/cell_assignments/{extra_dir}/{organelle}.csv"
            )
            t0 = time.time()

            # add column to organelle_df
            organelle_df["Nearest Canaliculi-Cell Contact Distance (nm)"] = np.nan
            organelle_df["Nearest Sinusoid-Cell Contact Distance (nm)"] = np.nan

            # build one sub-DF per cell, skipping cell 0
            sub_dfs = [
                grp.copy()
                for cell_id, grp in organelle_df.groupby("Cell ID", sort=False)
                if cell_id != 0
            ]
            del organelle_df

            # run in parallel
            with parallel_backend("multiprocessing", n_jobs=-1):
                results = Parallel()(
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
                f"{analysis_dir}/{dataset}/cell_assignments/polarity/{extra_dir}",
                exist_ok=True,
            )
            organelle_df.to_csv(
                f"{analysis_dir}/{dataset}/cell_assignments/polarity/{extra_dir}/{organelle}.csv",
                index=False,
            )
            print(
                f"Processed dataset {dataset}, organelle: {organelle} in {time.time() - t0:.2f} seconds."
            )
