"""Quick inspection of the external ref_grm.rel files supplied by the user."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
SRC = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf/external_data")
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

ids = pd.read_csv(SRC / "ref_grm.rel.id", sep="\t", header=None, names=["FID", "IID"])
n = len(ids)
print(f"GRM matrix: {n} × {n} samples")

# Read lower-triangular .rel
G = np.zeros((n, n))
with open(SRC / "ref_grm.rel") as f:
    for i, line in enumerate(f):
        vals = [float(x) for x in line.split()]
        assert len(vals) == i + 1, f"row {i} has {len(vals)} values"
        G[i, :i + 1] = vals
        G[:i + 1, i] = vals
diag = np.diag(G)
off = G[np.triu_indices(n, k=1)]
print(f"Diagonal:     min={diag.min():.3f}  max={diag.max():.3f}  mean={diag.mean():.3f}")
print(f"Off-diagonal: min={off.min():.3f}   max={off.max():.3f}   mean={off.mean():.4f}")
print(f"Off-diagonal > 0.3: {(off > 0.3).sum()} pairs  (likely 1st-degree relatives or duplicates)")
print(f"Off-diagonal 0.1–0.3: {((off > 0.1) & (off <= 0.3)).sum()} pairs")

# Merge with our existing reference metadata to colour rows by superpop
ref_meta = pd.read_csv(ROOT / "student_reference_metadata.tsv", sep="\t")
labels = ids.merge(ref_meta, left_on="IID", right_on="sample_id", how="left")
print("\nSuperpop counts in this GRM:")
print(labels["superpopulation"].value_counts())

# Order samples by superpop then by population
order = labels.sort_values(["superpopulation", "population", "IID"]).index.to_numpy()
Go = G[np.ix_(order, order)]
labels_o = labels.iloc[order].reset_index(drop=True)

# --- Heatmap ---
fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0a0d12")
ax.set_facecolor("#0a0d12")
# Cap colour range for clarity
vmin, vmax = -0.02, 0.30
im = ax.imshow(Go, cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
# Superpop separators
SP_COLOR = {"AFR": "#e41a1c", "EUR": "#377eb8", "EAS": "#4daf4a",
            "SAS": "#984ea3", "AMR": "#ff7f00"}
prev = None
for i, sp in enumerate(labels_o["superpopulation"]):
    if prev is not None and sp != prev:
        ax.axhline(i - 0.5, color="white", lw=0.6, alpha=0.7)
        ax.axvline(i - 0.5, color="white", lw=0.6, alpha=0.7)
    prev = sp
# Coloured ticks at superpop centres
sp_groups = labels_o.groupby("superpopulation", sort=False).indices
xticks, xticklabels, tick_colors = [], [], []
for sp, idx in sp_groups.items():
    mid = (idx[0] + idx[-1]) / 2
    xticks.append(mid); xticklabels.append(sp); tick_colors.append(SP_COLOR.get(sp, "#999"))
ax.set_xticks(xticks); ax.set_xticklabels(xticklabels, color="white")
ax.set_yticks(xticks); ax.set_yticklabels(xticklabels, color="white")
for tl, c in zip(ax.get_xticklabels() + ax.get_yticklabels(), tick_colors * 2):
    tl.set_color(c); tl.set_fontweight("bold")
ax.set_title("External GRM (ref_grm.rel) — 208 1000G refs, ordered by superpopulation",
             color="white", fontsize=13, pad=14, fontweight="light")
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("GRM value (≈ 2 × kinship)", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.get_yticklabels(), color="white")
for s in ax.spines.values(): s.set_color("#2a313e")
fig.tight_layout()
out = OUT / "external_grm_heatmap.png"
fig.savefig(out, dpi=150, facecolor="#0a0d12", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out}")

# --- Sanity: mean GRM within vs between superpops ---
print("\nMean GRM by superpop pair:")
sp_list = sorted(labels["superpopulation"].dropna().unique())
mat = pd.DataFrame(index=sp_list, columns=sp_list, dtype=float)
for sp1 in sp_list:
    for sp2 in sp_list:
        i1 = labels[labels["superpopulation"] == sp1].index.to_numpy()
        i2 = labels[labels["superpopulation"] == sp2].index.to_numpy()
        sub = G[np.ix_(i1, i2)]
        # exclude diagonal when sp1==sp2
        if sp1 == sp2:
            iu = np.triu_indices(len(i1), k=1)
            v = sub[iu].mean() if len(iu[0]) else float("nan")
        else:
            v = sub.mean()
        mat.loc[sp1, sp2] = v
print(mat.round(4).to_string())
