# https://github.com/janelia-cellmap/project-tracking/issues/551#issuecomment-2914211913
# https://github.com/janelia-cellmap/project-tracking/issues/570#issuecomment-2914214164
# %%
import pandas as pd
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

data_dir = cfg["DATA_DIR"]
analysis_dir = cfg["ANALYSIS_DIR"]

# %%
# Only keep organelles that are part of cells, so remove those ids and write them out to new files. This will be used for relabeling.
import pandas as pd
from cellmap_analyze.util.image_data_interface import ImageDataInterface

datasets = ["jrc_mus-liver-zon-1", "jrc_mus-liver-zon-2"]
organelles = [
    "mito",
    "ld",
    "perox",
    "nuc",
    "er",
]  # , "mito", "ld", "perox", "nuc"]
for ds in datasets:
    for organelle in organelles:
        df = pd.read_csv(
            f"{analysis_dir}/{ds}/for_relabeling/cell_assignments/original_{organelle}_assigned_to_containing_cell.csv"
        )
        initial_length = len(df)
        # only keep those with "Cell ID" not equal to 0
        unassigned = df[df["Cell ID"] == 0]

        df = df[df["Cell ID"] != 0]
        df.to_csv(
            f"{analysis_dir}/{ds}/for_relabeling/cell_assignments/original_{organelle}_assigned_to_containing_cell_unassigned_removed.csv",
            index=False,
        )
        # print delta length
        try:
            voxel_size = ImageDataInterface(
                f"{data_dir}/{ds}/{ds}.zarr/original_{organelle}/s0"
            ).voxel_size
        except:
            voxel_size = ImageDataInterface(
                f"{data_dir}/{ds}/{ds}.zarr/original_{organelle}"
            ).voxel_size
        try:
            last_id = unassigned.iloc[-1]["Object ID"]
            last_one = unassigned.iloc[-1][
                ["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]
            ].to_list()
        except:
            last_id = None
            last_one = None
        print(
            f"{ds} {organelle} {voxel_size}, ratio: {len(df)/initial_length}, delta length: {initial_length - len(df)},{last_id}:{last_one}"
        )


# %%
# Rewrite updated ids, ie only the ones that are within cells
import pandas as pd
import time

datasets = ["jrc_mus-liver-zon-1", "jrc_mus-liver-zon-2"]
organelles = ["er", "mito", "ld", "perox", "nuc"]
for ds in datasets:
    for organelle in organelles:
        t = time.time()
        print(f"Rewriting {organelle} for {ds}")
        df = pd.read_csv(
            f"{analysis_dir}/{ds}/for_relabeling/cell_assignments/original_{organelle}_assigned_to_containing_cell_unassigned_removed.csv"
        )
        # rewrite objects ids starting at 1 for non-er:
        if organelle != "er":
            df["Object ID"] = list(range(1, len(df) + 1))
        df.to_csv(
            f"{analysis_dir}/{ds}/cell_assignments/{organelle}.csv",
            index=False,
        )
        print(f"Rewrote {organelle} for {ds} in {time.time() - t:.2f} seconds")

        if organelle == "mito":
            t = time.time()
            print(f"Rewriting mito_mem for {ds}")
            # also write out mito_mem, which is the same as mito cell assignments
            mito_mem_df = pd.read_csv(f"{analysis_dir}/{ds}/mito_mem.csv")
            # create a cell assignment column
            mito_mem_df["Cell ID"] = mito_mem_df["Object ID"].map(
                df.set_index("Object ID")["Cell ID"]
            )
            mito_mem_df.to_csv(
                f"{analysis_dir}/{ds}/cell_assignments/mito_mem.csv",
                index=False,
            )
            print(f"Rewrote mito_mem for {ds} in {time.time() - t:.2f} seconds")
