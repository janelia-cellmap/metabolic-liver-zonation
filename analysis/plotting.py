# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yaml
import time

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

data_dir = cfg["DATA_DIR"]
analysis_dir = cfg["ANALYSIS_DIR"]
skeleton_dir = cfg["SKELETON_DIR"]

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
if "mito_with_skeleton" in organelles:
    properties = [
        "radius mean (nm)",
        "lsp (nm)",
        "Volume (nm^3)",
        "Sphericity",
        "Approximate Sphere Fraction",
    ]
else:
    properties = ["Volume (nm^3)", "Sphericity", "Approximate Sphere Fraction"]

errors = [("std", "±1 Std Dev"), ("ci", "95% CI")]

# total number of rows = 4 props * 2 error types + 1 count plot + 1 ratio plot
n_rows = len(properties) * len(errors) + 1 + 1
n_cols = len(organelles)
for dataset in ["jrc_mus-liver-zon-1", "jrc_mus-liver-zon-2"]:

    fig, axes = plt.subplots(
        n_rows, len(organelles), figsize=(6 * n_cols, 4 * n_rows), sharex=True
    )

    for col, organelle in enumerate(organelles):
        t0 = time.time()
        print(f"Processing dataset {dataset}, organelle: {organelle}...")
        extra_dir = ""
        if organelle.endswith("_contacts"):
            extra_dir = "contact_sites/"

        # --- load & prepare your DataFrame as before ---
        # if dataset == "jrc_mus-liver-zon-1":
        #     csv_path = f"{analysis_dir}/jrc_mus-liver-zon-1/20250430_duplicate_ids/{extra_dir}{organelle}.csv"
        # else:
        cell_df = pd.read_csv(f"{analysis_dir}/{dataset}/cell_assignments/cell.csv")
        organelle_df = pd.read_csv(
            f"{analysis_dir}/{dataset}/cell_assignments/polarity/{extra_dir}/{organelle}.csv"
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

        # compute the size in each dimension
        dx = organelle_df["MAX X (nm)"] - organelle_df["MIN X (nm)"]
        dy = organelle_df["MAX Y (nm)"] - organelle_df["MIN Y (nm)"]
        dz = organelle_df["MAX Z (nm)"] - organelle_df["MIN Z (nm)"]

        # add a new column with the diagonal
        approximate_sphere_radius = np.sqrt(dx**2 + dy**2 + dz**2) / 2
        organelle_df["Approximate Sphere Fraction"] = organelle_df["Volume (nm^3)"] / (
            4 / 3 * np.pi * approximate_sphere_radius**3
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
        # org_df_for_writing.to_csv(
        #     f"{analysis_dir}/{dataset}/20250430_duplicate_ids/forDaniel/{organelle}.csv",
        #     index=False,
        # )

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
        print(
            f"Processed dataset {dataset}, organelle: {organelle} in {time.time() - t0:.2f} seconds"
        )
        break
    plt.tight_layout()
    plt.show()
    # %%
    # get smallest 50 where their Volume (nm^3) >= 1E10
    large_volumes = organelle_df[organelle_df["Volume (nm^3)"] >= 1e10]
    display(
        large_volumes.sort_values("Approximate Sphere Fraction")
        .head(50)["Object ID"]
        .to_list()
    )
    display(
        large_volumes.sort_values("Approximate Sphere Fraction")
        .tail(50)["Object ID"]
        .to_list()
    )
    organelle_df = large_volumes
# %%
# plot meshes
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trimesh
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

# --- User parameters ---
dataset = "jrc_mus-liver-zon-2"
mesh_dir = (
    f"{data_dir}/../new_meshes/meshes/single_resolution/{dataset}/mito/meshes"
)
grid_size = 10  # e.g. a 20×20 grid
thumb_px = 256  # resolution of each square thumbnail
max_frac = 1.0  # largest thumbnail fills its cell

# organelle_df must already be in scope, with columns:
#   'Object ID', 'lsp (nm)', 'Approximate Sphere Fraction'


# --- CPU‐only: mesh → aligned thumbnail + x‐extent ---
def mesh_to_thumbnail(ply_path, size=thumb_px):
    mesh = trimesh.load(ply_path, force="mesh")
    verts = mesh.vertices
    # 1) PCA to find principal axis
    centered = verts - verts.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, np.argmax(evals)]
    # 2) build transform aligning v → X‐axis
    target = np.array([1.0, 0.0, 0.0])
    axis = np.cross(v, target)
    if np.linalg.norm(axis) < 1e-8:
        T = np.eye(4)
    else:
        axis /= np.linalg.norm(axis)
        angle = np.arccos(
            np.dot(v, target) / (np.linalg.norm(v) * np.linalg.norm(target))
        )
        centroid = verts.mean(axis=0)
        T = trimesh.transformations.rotation_matrix(angle, axis, centroid)
    mesh_aligned = mesh.copy()
    mesh_aligned.apply_transform(T)
    # 3) project aligned mesh to XY for rendering
    verts2d = mesh_aligned.vertices[:, :2]
    tris2d = verts2d[mesh_aligned.faces]
    # 4) compute X‐extent for scaling using np.ptp
    x_extent = np.ptp(verts2d[:, 0])  # max–min along X
    # 5) render thumbnail
    fig = Figure(figsize=(size / 100, size / 100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.add_collection(
        PolyCollection(tris2d, facecolor="lightgrey", edgecolor="black", linewidths=0.1)
    )
    ax.set_xlim(verts2d[:, 0].min(), verts2d[:, 0].max())
    ax.set_ylim(verts2d[:, 1].min(), verts2d[:, 1].max())
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")  # keep aspect ratio
    fig.tight_layout(pad=0)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (size, size, 4)
    buf = buf[:, :, [1, 2, 3, 0]]  # ARGB → RGBA
    thumb = Image.fromarray(buf)
    return thumb, x_extent


# --- 1) Build 2D bins & centers in data‐space ---
def equal_count_edges(vals, grid_size):
    """
    Return grid_size+1 cut‐positions along 1D array `vals`
    such that each of the grid_size intervals
    contains as close to len(vals)/grid_size points as possible.
    """
    # the ideal quantiles
    qs = np.linspace(0, 1, grid_size+1)
    # get the quantiles; this may produce duplicates if data are tied
    edges = np.quantile(vals, qs, interpolation="linear")
    edges[0], edges[-1] = vals.min(), vals.max()
    # nudge any non‐strictly‐increasing edges apart
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = np.nextafter(edges[i-1], vals.max())
    return edges

# —— usage in your DataFrame ——
x = organelle_df["lsp (nm)"].values
y = organelle_df["Approximate Sphere Fraction"].values



x_edges = equal_count_edges(x, grid_size)
y_edges = equal_count_edges(y, grid_size)

# override centers to be equally spaced over the full range:
x_centers = np.linspace(x.min(), x.max(), grid_size)
y_centers = np.linspace(y.min(), y.max(), grid_size)
#x_edges = np.linspace(x.min(), x.max(), grid_size + 1)
#y_edges = np.linspace(y.min(), y.max(), grid_size + 1)
#x_centers = (x_edges[:-1] + x_edges[1:]) / 2
#y_centers = (y_edges[:-1] + y_edges[1:]) / 2

cell_w = (x_edges[-1] - x_edges[0])/grid_size
cell_h = (y_edges[-1] - y_edges[0])/grid_size

#organelle_df["x_bin"] = pd.cut(x, bins=x_edges, labels=False, include_lowest=True)
#organelle_df["y_bin"] = pd.cut(y, bins=y_edges, labels=False, include_lowest=True)

# --- 2) Sample one object per cell ---
sampled = (
    organelle_df.dropna(subset=["x_bin", "y_bin"])
    .groupby(["x_bin", "y_bin"], as_index=False)
    .apply(lambda g: g.sample(1))
    .reset_index(drop=True)
)

# --- 3) Render each mesh & record its X‐extent ---
thumbs = {}
x_extents = []
for _, row in tqdm(sampled.iterrows(), total=sampled.shape[0]):
    oid = int(row["Object ID"])
    ply = os.path.join(mesh_dir, f"{oid}.ply")
    if os.path.isfile(ply):
        thumb, extent = mesh_to_thumbnail(ply)
        thumbs[oid] = thumb
        x_extents.append(extent)
    else:
        thumbs[oid] = None
        x_extents.append(np.nan)

sampled["x_extent"] = x_extents
max_extent = sampled["x_extent"].dropna().max()


# map x_extent → fraction of cell width [0…max_frac]
def frac_of_cell(e):
    return (e / max_extent) * max_frac


# --- 4) Plot with precise extents so width ∝ real X‐extent ---
fig, ax = plt.subplots(figsize=(12, 12), dpi=150)

for _, row in tqdm(sampled.iterrows(), total=sampled.shape[0]):
    oid = int(row["Object ID"])
    thumb = thumbs.get(oid)
    ext = row["x_extent"]
    if thumb is None or np.isnan(ext):
        continue
    f = frac_of_cell(ext)
    w = cell_w * f
    h = cell_h * f
    i, j = int(row["x_bin"]), int(row["y_bin"])
    cx, cy = x_centers[i], y_centers[j]
    extent = [cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2]
    ax.imshow(thumb, extent=extent, aspect="auto")

# mark empty bins
all_bins = {(i, j) for i in range(grid_size) for j in range(grid_size)}
filled = set(zip(sampled.x_bin.astype(int), sampled.y_bin.astype(int)))
for i, j in sorted(all_bins - filled):
    ax.plot(x_centers[i], y_centers[j], "x", color="lightgray", alpha=0.5)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_xlabel("Longest Shortest Path (nm)")
ax.set_ylabel("Approximate Sphere Fraction")
ax.set_title("Mesh Grid — width ∝ longest point‐to‐point X‐extent")
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
