# %%
# load your local config (config.yaml)
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

data_dir = cfg["DATA_DIR"]
analysis_dir = cfg["ANALYSIS_DIR"]
skeleton_dir = cfg["SKELETON_DIR"]

import pandas as pd

for dataset in ["jrc_mus-liver-zon-1", "jrc_mus-liver-zon-2"]:
    general_mito_information_df = pd.read_csv(
        f"{analysis_dir}/{dataset}/cell_assignments/mito.csv"
    )

    skeleton_information_df = pd.read_csv(
        f"{skeleton_dir}/{dataset}/mito/metrics/skeleton_metrics.csv"
    )
    assert len(skeleton_information_df) == len(general_mito_information_df)
    # combine general information dataframe with skeleton dataframe columns based on ids
    # relabel skeleton df "id" with "Object ID"

    skeleton_information_df = skeleton_information_df.rename(
        columns={"id": "Object ID"}
    )
    combined_df = general_mito_information_df.merge(
        skeleton_information_df, on="Object ID"
    )
    # after your rename…
    # skeleton_information_df = skeleton_information_df.rename(
    #     columns={"id": "Object ID"}
    # )

    # find all Object IDs in mito.csv not in skeleton_metrics.csv
    # missing_ids = general_mito_information_df.loc[
    #     ~general_mito_information_df["Object ID"].isin(
    #         skeleton_information_df["Object ID"]
    #     ),
    #     "Object ID",
    # ]

    # print(f"Missing {len(missing_ids)} IDs:")
    # print(missing_ids.tolist())
    # break
    # sort by object id
    combined_df = combined_df.sort_values(by=["Object ID"])
    # reset index
    combined_df = combined_df.reset_index(drop=True)
    # write to csv and ignore index
    print(combined_df.loc[combined_df["num branches"].idxmax()])
    combined_df.to_csv(
        f"{analysis_dir}/{dataset}/cell_assignments/mito_with_skeleton.csv",
        index=False,
    )
    display(combined_df.describe())

# # %%
# # sort by minimum volume
# general_mito_information_df.sort_values(
#     by="Volume (nm^3)", ascending=True, inplace=True
# )
# # %%
# # find all those with volume < 2E7
# small_mito = general_mito_information_df[
#     general_mito_information_df["Volume (nm^3)"] < 2e7
# ]
# # %%
# import pandas as pd

# duplicated = pd.read_csv(
#     "{analysis_dir}/jrc_mus-liver-zon-1/duplicate_ids/for_relabeling/original_mito.csv"
# )
# deduplicated = pd.read_csv(
#     "{analysis_dir}/jrc_mus-liver-zon-1/for_relabeling/original_mito.csv"
# )
# # %%# only keep the unique COM‐triples in duplicated
# cols = ["COM X (nm)", "COM Y (nm)", "COM Z (nm)"]

# # 1) Create rounded‐to‐1nm versions of both DataFrames
# duplicated_rounded = duplicated.assign(**{c: duplicated[c].round(-3) for c in cols})
# deduplicated_rounded = deduplicated.assign(
#     **{c: deduplicated[c].round(-3) for c in cols}
# )

# # 2) Get unique rounded COMs from duplicated
# dup_unique = duplicated_rounded[cols].drop_duplicates()

# # 3) Merge on the rounded COMs, marking where there was no match
# merged = deduplicated_rounded.merge(dup_unique, on=cols, how="left", indicator=True)

# # 4) Select those deduplicated rows whose rounded COM didn’t find a partner
# not_in_dup = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
# # %%
# display(deduplicated[deduplicated["Object ID"] == 7612])
# display(duplicated[duplicated["Object ID"] == 7612])


# # %%
# for df in (duplicated, deduplicated):
#     # compute bounding‐box volume
#     bbox_vol = (
#         (df["MAX X (nm)"] - df["MIN X (nm)"])
#         * (df["MAX Y (nm)"] - df["MIN Y (nm)"])
#         * (df["MAX Z (nm)"] - df["MIN Z (nm)"])
#     )
#     # add ratio column
#     df["bbox_to_volume_ratio"] = bbox_vol / df["Volume (nm^3)"]
# # get deduplicated maximum bbox_to_volume_ratio row
# display(
#     deduplicated[
#         deduplicated["bbox_to_volume_ratio"]
#         == deduplicated["bbox_to_volume_ratio"].max()
#     ]
# )
# display(
#     duplicated[
#         duplicated["bbox_to_volume_ratio"] == duplicated["bbox_to_volume_ratio"].max()
#     ]
# )

# # %%
# import pandas as pd

# df = pd.read_csv(
#     "{analysis_dir}/jrc_mus-liver-zon-1/for_relabeling/cell_assignments/original_ld_assigned_to_containing_cell_unassigned_removed.csv"
# )
# bbox_vol = (
#     (df["MAX X (nm)"] - df["MIN X (nm)"])
#     * (df["MAX Y (nm)"] - df["MIN Y (nm)"])
#     * (df["MAX Z (nm)"] - df["MIN Z (nm)"])
# )
# # add ratio column
# df["bbox_to_volume_ratio"] = bbox_vol / df["Volume (nm^3)"]
# df[df["bbox_to_volume_ratio"] == df["bbox_to_volume_ratio"].max()]
# # %%
# df.describe()

# # %%

# %%
