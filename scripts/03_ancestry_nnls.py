"""Supervised global ancestry estimation via NNLS on superpop allele frequencies.

Model: for each student i, find non-negative ancestry weights alpha_ik (k = superpop)
such that  geno_ij ~ 2 * sum_k alpha_ik * p_kj   for all SNPs j,
subject to sum_k alpha_ik = 1, alpha_ik >= 0.

Solved as a constrained least-squares with simplex projection (active-set NNLS
plus normalization).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
WORK = ROOT / "work"
RES = ROOT / "results"

SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]

# Load freqs - merge into a single SNP x superpop matrix
freqs = []
for s in SUPERPOPS:
    df = pd.read_csv(WORK / f"freq_{s}.afreq", sep="\t")
    # Columns: #CHROM, ID, REF, ALT, PROVISIONAL_REF?, ALT_FREQS, OBS_CT
    df = df.rename(columns={"#CHROM": "CHROM", "ALT_FREQS": f"f_{s}"})
    if not freqs:
        keep = df[["ID", "REF", "ALT", f"f_{s}"]].copy()
    else:
        keep = keep.merge(df[["ID", f"f_{s}"]], on="ID")
    freqs.append(s)
freq_df = keep
print(f"Reference allele freq matrix: {freq_df.shape}")

# Load student genotype dosages
print("Loading student .raw ...")
raw = pd.read_csv(WORK / "stud_raw.raw", sep="\t")
# Columns: FID IID PAT MAT SEX PHENOTYPE then SNP_A1 columns
id_col = "IID" if "IID" in raw.columns else "#IID"
ids = raw[id_col].values
meta_cols = [c for c in raw.columns if c in {"FID", "IID", "#IID", "#FID", "PAT", "MAT", "SEX", "PHENOTYPE"}]
geno_cols = [c for c in raw.columns if c not in meta_cols]
print(f"Students: {len(ids)}, SNPs in .raw: {len(geno_cols)}")

# The .raw header has SNP_<counted_allele> form; counted_allele is ALT by default.
# Map column -> snp_id and ensure freqs aligned to same allele direction.
import re
snp_alt = {}
for c in geno_cols:
    m = re.match(r"^(.+)_(.+)$", c)
    if not m:
        continue
    snp_alt[c] = (m.group(1), m.group(2))

# Build aligned matrix
# Build dict of freq rows by ID
freq_by_id = freq_df.set_index("ID")
common_cols = []
flip = []
for c in geno_cols:
    sid, counted = snp_alt[c]
    if sid not in freq_by_id.index:
        continue
    row = freq_by_id.loc[sid]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    if counted == row["ALT"]:
        flip.append(False)
    elif counted == row["REF"]:
        flip.append(True)
    else:
        continue
    common_cols.append(c)
print(f"Common SNPs aligned: {len(common_cols)}")

geno = raw[common_cols].values.astype(np.float32)  # 70 x N
P = freq_by_id.loc[[snp_alt[c][0] for c in common_cols], [f"f_{s}" for s in SUPERPOPS]].values  # N x K
flip = np.array(flip)
# Flip frequencies where counted allele is REF
P_aligned = np.where(flip[:, None], 1 - P, P)

# 2*P_aligned is expected dosage if 100% that superpop
A_mat = 2 * P_aligned  # N x K

# Solve NNLS per student, then normalize to sum=1
results = []
for i, sid in enumerate(ids):
    y = geno[i]
    valid = ~np.isnan(y)
    A = A_mat[valid]
    yi = y[valid]
    alpha, _ = nnls(A, yi, maxiter=200)
    s = alpha.sum()
    if s > 0:
        alpha = alpha / s
    results.append([sid] + alpha.tolist())

anc = pd.DataFrame(results, columns=["sample_id"] + SUPERPOPS)
anc.to_csv(RES / "ancestry_proportions.tsv", sep="\t", index=False, float_format="%.4f")
print(f"\nSaved {RES/'ancestry_proportions.tsv'}")
print("\nFirst 15 students:")
print(anc.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# --- Plot ancestry barplot ---
# Sort by dominant ancestry, then by next-largest
def sort_key(row):
    arr = np.array([row[s] for s in SUPERPOPS])
    return (-arr.argmax(), -arr.max())
anc_sorted = anc.sort_values(by=SUPERPOPS, ascending=[False]*5).reset_index(drop=True)
# Better: sort by dominant superpop, then by its proportion
anc_sorted["dom"] = anc[SUPERPOPS].idxmax(axis=1)
anc_sorted["dom_val"] = anc[SUPERPOPS].max(axis=1)
anc_sorted = anc_sorted.sort_values(["dom", "dom_val"], ascending=[True, False]).reset_index(drop=True)

SUPERPOP_COLORS = {"AFR": "#e41a1c", "EUR": "#377eb8", "EAS": "#4daf4a",
                   "SAS": "#984ea3", "AMR": "#ff7f00"}

fig, ax = plt.subplots(figsize=(16, 6))
bottom = np.zeros(len(anc_sorted))
for s in SUPERPOPS:
    vals = anc_sorted[s].values
    ax.bar(range(len(anc_sorted)), vals, bottom=bottom,
           color=SUPERPOP_COLORS[s], label=s, width=0.95, edgecolor="none")
    bottom += vals
ax.set_xticks(range(len(anc_sorted)))
ax.set_xticklabels(anc_sorted["sample_id"].values, rotation=90, fontsize=7)
ax.set_ylabel("Ancestry proportion")
ax.set_xlim(-0.5, len(anc_sorted) - 0.5)
ax.set_ylim(0, 1)
ax.set_title("Global ancestry (supervised NNLS on 1000G superpopulation frequencies, 181k SNPs)")
ax.legend(loc="upper right", ncol=5, fontsize=10, framealpha=0.95)
fig.tight_layout()
fig.savefig(RES / "ancestry_barplot.png", dpi=150)
plt.close(fig)
print(f"Saved {RES/'ancestry_barplot.png'}")

# How many are admixed (no single ancestry > 0.9)?
admixed = anc[SUPERPOPS].max(axis=1) < 0.9
print(f"\nAdmixed (max ancestry < 0.9): {admixed.sum()} / {len(anc)}")
admixed_strict = anc[SUPERPOPS].max(axis=1) < 0.75
print(f"Strongly admixed (max < 0.75): {admixed_strict.sum()} / {len(anc)}")
