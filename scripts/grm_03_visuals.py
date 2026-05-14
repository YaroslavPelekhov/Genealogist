"""Solution B — visualizations.
  1. GRM heatmap (joint 278x278, ordered by superpop)
  2. Reference-projected PCA (PC1/2 + PC3/4)
  3. Comparison plot: GRM-PCA vs plink-PCA (Solution A)
  4. F-inbreeding distribution + outlier callout
  5. GRM-vs-KING scatter (student-student pairs)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
WORK = ROOT / "work"
RES = ROOT / "results"

SP_COLOR = {"AFR": "#e41a1c", "EUR": "#377eb8", "EAS": "#4daf4a",
            "SAS": "#984ea3", "AMR": "#ff7f00", "STUDENT": "#222"}

def load_rel(prefix):
    p = Path(prefix)
    with open(p.with_suffix(".rel.id")) as f:
        lines = [l.strip().split("\t") for l in f if l.strip() and not l.strip().startswith("#")]
    ids = [row[-1] for row in lines]
    n = len(ids)
    G = np.zeros((n, n))
    with open(p.with_suffix(".rel")) as f:
        for i, line in enumerate(f):
            vals = [float(x) for x in line.split()]
            G[i, :i+1] = vals
            G[:i+1, i] = vals
    return G, ids

G, ids = load_rel(WORK / "grm_joint")
n = len(ids)

# Annotate
ref_meta = pd.read_csv(ROOT / "student_reference_metadata.tsv", sep="\t").set_index("sample_id")
fam_meta = pd.read_csv(ROOT / "student_family_metadata.tsv", sep="\t").set_index("sample_id")
labels = pd.DataFrame({"IID": ids})
labels["is_student"] = ~labels["IID"].str.startswith("ref")
labels["superpopulation"] = labels["IID"].map(
    lambda i: ref_meta.loc[i, "superpopulation"] if i in ref_meta.index else "STUDENT")
labels["population"] = labels["IID"].map(
    lambda i: ref_meta.loc[i, "population"] if i in ref_meta.index else "STUDENT")
labels["age"] = labels["IID"].map(lambda i: fam_meta.loc[i, "age"] if i in fam_meta.index else np.nan)

# --- 1. GRM heatmap (joint, ordered by superpop) ---
SP_ORDER = ["AFR", "AMR", "EAS", "EUR", "SAS", "STUDENT"]
labels["sp_ord"] = labels["superpopulation"].map({s: i for i, s in enumerate(SP_ORDER)})
order = labels.sort_values(["sp_ord", "population", "IID"]).index.to_numpy()
G_o = G[np.ix_(order, order)]
labels_o = labels.iloc[order].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(13, 11), facecolor="#0a0d12")
ax.set_facecolor("#0a0d12")
im = ax.imshow(G_o, cmap="magma", vmin=-0.05, vmax=0.30, aspect="equal")
# Group separators
prev = None
group_bounds = []
for i, sp in enumerate(labels_o["superpopulation"]):
    if prev is not None and sp != prev:
        ax.axhline(i - 0.5, color="white", lw=0.7, alpha=0.7)
        ax.axvline(i - 0.5, color="white", lw=0.7, alpha=0.7)
        group_bounds.append((prev, i))
    prev = sp
group_bounds.append((prev, len(labels_o)))
# Centered labels per group
last_start = 0
xticks, xticklabels, xcolors = [], [], []
for gname, gend in group_bounds:
    xticks.append((last_start + gend - 1) / 2)
    xticklabels.append(gname)
    xcolors.append(SP_COLOR.get(gname, "#fff"))
    last_start = gend
ax.set_xticks(xticks); ax.set_yticks(xticks)
ax.set_xticklabels(xticklabels); ax.set_yticklabels(xticklabels)
for tl, c in zip(ax.get_xticklabels() + ax.get_yticklabels(), xcolors * 2):
    tl.set_color(c); tl.set_fontweight("bold")
ax.set_title("Joint GRM (278 × 278) — refs + students, ordered by superpopulation\n"
             "[Solution B — VanRaden 2008 normalization]",
             color="white", fontsize=13, pad=14, fontweight="light")
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("GRM value (≈ 2 × kinship + structure)", color="white")
cbar.ax.tick_params(colors="white")
plt.setp(cbar.ax.get_yticklabels(), color="white")
for s in ax.spines.values(): s.set_color("#2a313e")
fig.tight_layout()
fig.savefig(RES / "grm_joint_heatmap.png", dpi=150, facecolor="#0a0d12", bbox_inches="tight")
plt.close(fig)
print("Saved grm_joint_heatmap.png")

# --- 2. Reference-projected PCA ---
proj = pd.read_csv(RES / "grm_projected_pca.tsv", sep="\t")
proj["superpopulation"] = proj["IID"].map(
    lambda i: ref_meta.loc[i, "superpopulation"] if i in ref_meta.index else "STUDENT")
proj["population"] = proj["IID"].map(
    lambda i: ref_meta.loc[i, "population"] if i in ref_meta.index else "STUDENT")

fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor="#0a0d12")
for ax, (xc, yc) in zip(axes, [("PC1","PC2"), ("PC3","PC4")]):
    ax.set_facecolor("#0a0d12")
    for sp in ["AFR", "AMR", "EAS", "EUR", "SAS"]:
        sub = proj[(~proj["is_student"]) & (proj["superpopulation"] == sp)]
        ax.scatter(sub[xc], sub[yc], s=44, c=SP_COLOR[sp], alpha=0.7,
                   edgecolors="none", label=sp)
    studs = proj[proj["is_student"]]
    ax.scatter(studs[xc], studs[yc], s=110, marker="X", c="white",
               edgecolors="black", linewidths=1.2, label="students (projected)")
    ax.set_xlabel(xc, color="white"); ax.set_ylabel(yc, color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#2a313e")
    ax.grid(alpha=0.1, color="white")
axes[0].legend(loc="best", fontsize=9, facecolor="#0e1116", edgecolor="#2a313e",
               labelcolor="white")
fig.suptitle("GRM-based PCA: students PROJECTED onto reference axes (LASER-style)\n"
             "[Solution B — students do NOT influence the principal axes]",
             color="white", fontsize=13, y=1.0, fontweight="light")
fig.tight_layout()
fig.savefig(RES / "grm_projected_pca.png", dpi=150, facecolor="#0a0d12", bbox_inches="tight")
plt.close(fig)
print("Saved grm_projected_pca.png")

# --- 3. Comparison: GRM projected PCA vs Solution-A joint PCA ---
sol_a = pd.read_csv(RES / "pca_coords.tsv", sep="\t")
# Align on IID
sol_a_keep = sol_a[["IID", "PC1", "PC2", "PC3", "PC4"]].rename(
    columns={c: f"A_{c}" for c in ["PC1","PC2","PC3","PC4"]})
sol_b_keep = proj[["IID", "PC1", "PC2", "PC3", "PC4", "superpopulation", "is_student"]].rename(
    columns={c: f"B_{c}" for c in ["PC1","PC2","PC3","PC4"]})
cmp = sol_a_keep.merge(sol_b_keep, on="IID")
# Flip PC signs if necessary (eigenvector direction is arbitrary)
for k in [1, 2, 3, 4]:
    r = np.corrcoef(cmp[f"A_PC{k}"], cmp[f"B_PC{k}"])[0, 1]
    if r < 0:
        cmp[f"B_PC{k}"] *= -1

fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0a0d12")
for ax, k in zip(axes, [1, 2, 3, 4]):
    ax.set_facecolor("#0a0d12")
    for sp, sub in cmp.groupby("superpopulation"):
        ax.scatter(sub[f"A_PC{k}"], sub[f"B_PC{k}"], s=24, c=SP_COLOR.get(sp, "#999"),
                   alpha=0.7, edgecolors="none", label=sp if k == 1 else None)
    r = np.corrcoef(cmp[f"A_PC{k}"], cmp[f"B_PC{k}"])[0, 1]
    ax.set_xlabel(f"Solution A — PC{k} (plink --pca)", color="white")
    ax.set_ylabel(f"Solution B — PC{k} (GRM eig)", color="white")
    ax.set_title(f"PC{k}  ·  r = {r:.4f}", color="white", fontsize=11)
    ax.tick_params(colors="white"); ax.grid(alpha=0.1, color="white")
    for s in ax.spines.values(): s.set_color("#2a313e")
axes[0].legend(loc="best", fontsize=8, facecolor="#0e1116", edgecolor="#2a313e",
               labelcolor="white")
fig.suptitle("Solution A vs Solution B principal components",
             color="white", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(RES / "grm_vs_plink_pca.png", dpi=150, facecolor="#0a0d12", bbox_inches="tight")
plt.close(fig)
print("Saved grm_vs_plink_pca.png")

# --- 4. F_inbreeding ---
F = pd.read_csv(RES / "grm_F_inbreeding.tsv", sep="\t")
fig, ax = plt.subplots(figsize=(13, 5), facecolor="#0a0d12")
ax.set_facecolor("#0a0d12")
F["dom_group"] = F["superpopulation"]
F_sorted = F.sort_values(["dom_group", "F_inbreeding"]).reset_index(drop=True)
for i, (_, r) in enumerate(F_sorted.iterrows()):
    c = SP_COLOR.get(r["dom_group"], "#999")
    ax.bar(i, r["F_inbreeding"], color=c, width=1.0, edgecolor="none")
# Annotate top outliers
top = F_sorted.iloc[F_sorted["F_inbreeding"].abs().sort_values(ascending=False).head(8).index]
for i, r in top.iterrows():
    ax.annotate(r["sample_id"], (i, r["F_inbreeding"]),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=6, color="white")
ax.axhline(0, color="white", lw=0.4, alpha=0.4)
# Group separators + group labels
prev = None
group_start = 0
for i, sp in enumerate(F_sorted["dom_group"]):
    if prev is not None and sp != prev:
        ax.axvline(i - 0.5, color="#666", lw=0.5, alpha=0.5)
        ax.text((group_start + i - 1) / 2, ax.get_ylim()[1] * 0.95, prev,
                ha="center", color=SP_COLOR.get(prev, "white"),
                fontsize=11, fontweight="bold")
        group_start = i
    prev = sp
ax.text((group_start + len(F_sorted) - 1) / 2, ax.get_ylim()[1] * 0.95, prev,
        ha="center", color=SP_COLOR.get(prev, "white"), fontsize=11, fontweight="bold")
ax.set_xlim(-0.5, len(F_sorted) - 0.5)
ax.set_ylabel("F = G_ii − 1", color="white")
ax.tick_params(axis="y", colors="white")
ax.set_xticks([])
ax.set_title("Per-sample inbreeding coefficient from GRM diagonal\n"
             "(Warning: not corrected for population stratification — biased upward for AFR/EUR)",
             color="white", fontsize=12, fontweight="light")
for s in ax.spines.values(): s.set_color("#2a313e")
fig.tight_layout()
fig.savefig(RES / "grm_F_inbreeding.png", dpi=150, facecolor="#0a0d12", bbox_inches="tight")
plt.close(fig)
print("Saved grm_F_inbreeding.png")

# --- 5. GRM vs KING (student-student pairs) ---
cmp_pairs = pd.read_csv(RES / "grm_vs_king_pairs.tsv", sep="\t")
fig, ax = plt.subplots(figsize=(8.5, 8), facecolor="#0a0d12")
ax.set_facecolor("#0a0d12")
ax.scatter(cmp_pairs["KING"], cmp_pairs["GRM"], s=14, c="#7cd0ff",
           alpha=0.6, edgecolors="none", label=f"{len(cmp_pairs)} pairs")
ax.axhline(0.0442, color="#ffb74d", lw=0.7, alpha=0.6, linestyle="--",
           label="GRM threshold 0.044")
ax.axvline(0.0442, color="#7be57b", lw=0.7, alpha=0.6, linestyle="--",
           label="KING threshold 0.044")
ax.axhline(0.177, color="#f93838", lw=0.5, alpha=0.5)
ax.axvline(0.177, color="#f93838", lw=0.5, alpha=0.5)
ax.text(0.18, 0.31, "1st-degree", color="#f93838", fontsize=10)
ax.text(0.18, 0.07, "GRM-2nd zone", color="#ffb74d", fontsize=10)
ax.plot([-0.05, 0.55], [-0.05, 0.55], color="#666", lw=0.5, linestyle=":", alpha=0.6,
        label="y = x")
r = np.corrcoef(cmp_pairs["KING"], cmp_pairs["GRM"])[0, 1]
ax.set_xlabel("KING kinship (Solution A)", color="white")
ax.set_ylabel("GRM value (Solution B)", color="white")
ax.set_title(f"GRM vs KING — 70 students × 69 / 2 = {len(cmp_pairs)} pairs\n"
             f"Pearson r = {r:.3f}",
             color="white", fontsize=13, fontweight="light")
ax.tick_params(colors="white"); ax.grid(alpha=0.1, color="white")
for s in ax.spines.values(): s.set_color("#2a313e")
ax.legend(loc="upper left", fontsize=9, facecolor="#0e1116", edgecolor="#2a313e",
          labelcolor="white")
fig.tight_layout()
fig.savefig(RES / "grm_vs_king_scatter.png", dpi=150, facecolor="#0a0d12", bbox_inches="tight")
plt.close(fig)
print("Saved grm_vs_king_scatter.png")
