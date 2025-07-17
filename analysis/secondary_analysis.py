# %%
import pandas as pd
import yaml
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import ast

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
analysis_dir = cfg["ANALYSIS_DIR"]

# compiled all results in f"{analysis_dir}/{dataset}/fundamental_results" by symlinking:
# NOTE: where we didnt explicitly do cell_assignments and polarity assignment, they came from the base directory, otherwise came from polarity directory
# canaliculi_cell_contacts.csv -> ../contact_sites/canaliculi_cell_contacts.csv
# er.csv -> ../cell_assignments/polarity/er.csv
# er_ld_contacts.csv -> ../cell_assignments/polarity/contact_sites/er_ld_contacts.csv
# er_mito_contacts.csv -> ../cell_assignments/polarity/contact_sites/er_mito_contacts.csv
# er_perox_contacts.csv -> ../cell_assignments/polarity/contact_sites/er_perox_contacts.csv
# ld.csv -> ../cell_assignments/polarity/ld.csv
# ld_perox_contacts.csv -> ../cell_assignments/polarity/contact_sites/ld_perox_contacts.csv
# mito.csv -> ../cell_assignments/polarity/mito_with_skeleton.csv
# mito_ld_contacts.csv -> ../cell_assignments/polarity/contact_sites/mito_ld_contacts.csv
# mito_mem.csv -> ../mito_mem.csv
# mito_perox_contacts.csv -> ../cell_assignments/polarity/contact_sites/mito_perox_contacts.csv
# nuc.csv -> ../cell_assignments/polarity/nuc.csv
# perox.csv -> ../cell_assignments/polarity/perox.csv
# sinusoid_cell_contacts.csv -> ../contact_sites/sinusoid_cell_contacts.csv

# Function to calculate SA/V ratio sphericity and MCI (Morphological Complexity Index) square
# MCI_square = 1 if the shape fit a perfect sphere. Any shape more complex than a sphere (e.g., branched mitochondrion) will have MCI_square > 1.
def add_new_metrics(df, cell_df=None):
    surface_area = df["Surface Area (nm^2)"]
    volume = df["Volume (nm^3)"]
    df["Surface Area/Volume (1/nm)"] = surface_area / volume
    df["Sphericity"] = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / surface_area
    df["MCI_square (1/nm^2)"] = (surface_area ** 2) / (16 * (np.pi ** 2) * (volume ** 2))

    dx = df["MAX X (nm)"] - df["MIN X (nm)"]
    dy = df["MAX Y (nm)"] - df["MIN Y (nm)"]
    dz = df["MAX Z (nm)"] - df["MIN Z (nm)"]

    # add a new column with the diagonal
    approximate_sphere_radius = np.sqrt(dx**2 + dy**2 + dz**2) / 2
    df["Approximate Sphere Fraction"] = df["Volume (nm^3)"] / (
        4 / 3 * np.pi * approximate_sphere_radius**3
    )

def add_distance_to_cell_com(df, cell_df):
    # 1) build mapping Series for each coordinate
    map_x = cell_df.set_index('Object ID')['COM X (nm)']
    map_y = cell_df.set_index('Object ID')['COM Y (nm)']
    map_z = cell_df.set_index('Object ID')['COM Z (nm)']

    # 2) map cell‐COM into df, in place
    df['cell_COM X (nm)'] = df['Cell ID'].map(map_x)
    df['cell_COM Y (nm)'] = df['Cell ID'].map(map_y)
    df['cell_COM Z (nm)'] = df['Cell ID'].map(map_z)

    # 3) compute in‐place Euclidean distance
    df['Distance to Cell COM (nm)'] = np.sqrt(
        (df['COM X (nm)']  - df['cell_COM X (nm)'])**2 +
        (df['COM Y (nm)']  - df['cell_COM Y (nm)'])**2 +
        (df['COM Z (nm)']  - df['cell_COM Z (nm)'])**2
    )

    # 4) clean up (if you don’t need the mapped cols afterwards)
    df.drop(
        ['cell_COM X (nm)', 'cell_COM Y (nm)', 'cell_COM Z (nm)'],
        axis=1,
        inplace=True
    )

classes = [
    # "er",
    # "mito",
    "ld",
    # "perox",
    # "nuc",
    # "er_mito_contacts",
    # "er_perox_contacts",
    # "er_ld_contacts",
    # "ld_perox_contacts",
    # "mito_perox_contacts",
    "mito_ld_contacts",
    "canaliculi_cell_contacts",
    "sinusoid_cell_contacts",
    "cell"

]

def process_class(dataset: str, analysis_dir: str, current_class: str):
    """
    Reads a CSV for a given class, applies new metrics, and returns the class name and DataFrame.
    """
    input_dir = os.path.join(analysis_dir, dataset, "fundamental_results")
    file_path = os.path.join(input_dir, f"{current_class}.csv")
    df = pd.read_csv(file_path)
    add_new_metrics(df)
    return current_class, df


def parallel_read(dataset: str, analysis_dir: str):
    """
    Reads all class CSVs for a dataset in parallel using all available CPUs,
    displaying progress, and returns a dict mapping class names to DataFrames.
    """
    output_dir = os.path.join(analysis_dir, dataset, "secondary_results")
    os.makedirs(output_dir, exist_ok=True)

    data_dict = {}
    num_workers = os.cpu_count() or 1
    # Use a process pool with all available CPUs
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_class, dataset, analysis_dir, cls): cls
            for cls in classes
        }
        # Iterate over completed futures with a progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing classes for {dataset}"):
            cls_name, df = future.result()
            data_dict[cls_name] = df

    return data_dict

def summarize_contacts(contacts_df, organelle_key, contacting_organelle_key):
    duplicate_df = contacts_df.copy()
    surface_area_col = f"Contacting {organelle_key} Surface Area (nm^2)"
    ids_col = f"Contacting {organelle_key} IDs"
    contacting_ids_col = "Contacting " + contacting_organelle_key + " IDs"

    duplicate_df[ids_col] = duplicate_df[ids_col].str.replace(r'np\.int64\((\d+)\)', r'\1', regex=True).apply(ast.literal_eval)
    duplicate_df[surface_area_col] = duplicate_df[surface_area_col].str.replace(r'np\.int64\((\d+)\)', r'\1', regex=True).apply(ast.literal_eval)

    # Explode lists into separate rows
    exploded_df = duplicate_df[["Object ID", ids_col, surface_area_col, contacting_ids_col]].explode([ids_col, surface_area_col])
    # Drop rows where ID is NaN
    exploded_df = exploded_df.dropna(subset=[ids_col])

    exploded_df[ids_col] = exploded_df[ids_col].astype(int)
    exploded_df[surface_area_col] = exploded_df[surface_area_col].astype(float)
    
    # Group by organelle ID to calculate summary metrics
    summary = exploded_df.groupby(ids_col).agg(
        total_surface_area=pd.NamedAgg(column=surface_area_col, aggfunc="sum"),
        #total_contacts=pd.NamedAgg(column="Object ID", aggfunc="count"),
        #total_unique_objects=pd.NamedAgg(column="Contacting " + contacting_organelle_key + " IDs", aggfunc=lambda x: x.nunique())
    ).reset_index().rename(columns={ids_col: "Object ID"})
    # rename column
    summary = summary.rename(columns={"total_surface_area": f"Surface Area Touching {contacting_organelle_key} (nm^2)"})
    #summary = summary.rename(columns={"Organelle ID": f"Object ID"})
    #summary["Organelle Type"] = organelle_key
    return summary

def add_cell_object_stats(df, cell_df, organelle_name):
    """
    Adds two columns to cell_df in place:
      - out_count_col:  number of rows in df whose Cell ID == each Object ID in cell_df
      - out_volume_col: sum of their volume_col
    
    Parameters
    ----------
    df : pandas.DataFrame
      must contain cell_id_col and volume_col
    cell_df : pandas.DataFrame
      must contain obj_id_col (the cell’s ID)
    """
    obj_id_col='Object ID'
    cell_id_col='Cell ID'
    volume_col='Volume (nm^3)'
    out_count_col=f'Number of {organelle_name}'
    out_volume_col=f'Total {organelle_name} Volume (nm^3)'
    # 1) compute grouped stats on df
    grp = df.groupby(cell_id_col).agg(
        **{out_count_col: (obj_id_col, 'count'),
           out_volume_col: (volume_col, 'sum')}
    )
    # 2) map them onto cell_df (filling 0 where no objects)
    cell_df[out_count_col] = cell_df[obj_id_col].map(grp[out_count_col]).fillna(0).astype(int)
    cell_df[out_volume_col] = cell_df[obj_id_col].map(grp[out_volume_col]).fillna(0.0)


def update_dataframe_with_contact_information(data, organelle_key, contact_site_list):
    for contact_site in contact_site_list:
        contacting_organelle_key = contact_site.replace("_","").replace("contacts", "").replace(organelle_key,"")
        contact_site_summary = summarize_contacts(data[contact_site], organelle_key, contacting_organelle_key)
    # merge contact site summary with data[organelle key] on Object ID
        data[organelle_key] = data[organelle_key].merge(contact_site_summary, on="Object ID", how="left")

datasets = ["jrc_mus-liver-zon-1", "jrc_mus-liver-zon-2"]

# Process each dataset in sequence or in parallel if desired
for dataset in datasets:
    data = parallel_read(dataset, analysis_dir)
    # sort data columns by what is in classes
    data = {k: v for k, v in sorted(data.items(), key=lambda item: classes.index(item[0]) if item[0] in classes else len(classes))}
    # for cell contacts, we want to get the amount of surface area of cells for canaliculi, sinusoid
    update_dataframe_with_contact_information(data, "cell", ["canaliculi_cell_contacts", "sinusoid_cell_contacts"]) 
    # write out data to output_dir
    output_dir = os.path.join(analysis_dir, dataset, "secondary_results")
    os.makedirs(output_dir, exist_ok=True)
    for key, df in data.items():
        if "cell" not in key:
            add_distance_to_cell_com(df, data["cell"])
            add_cell_object_stats(df, data["cell"], key)
            
        output_path = os.path.join(output_dir, f"{key}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {key} data to {output_path}")
    # break  # remove or adjust as needed

# # %%
# # Define a function to summarize contact information


# # Calculate for ER
# import time
# start_time = time.time()
# canaliculi_cell_summary = summarize_contacts(data["canaliculi_cell_contacts"], "cell", "canaliculi")
# sinusoid_cell_summary = summarize_contacts(data["sinusoid_cell_contacts"], "cell", "sinusoid")

# # Results

# # %%
# summarize_contacts(data["er_ld_contacts"], "ld","er")
# # %%

# %%
