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
