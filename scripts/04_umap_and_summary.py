"""UMAP visualization and final summary table per student."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
RES = ROOT / "results"

pca = pd.read_csv(RES / "pca_coords.tsv", sep="\t")
pred = pd.read_csv(RES / "population_predictions.tsv", sep="\t")
anc = pd.read_csv(RES / "ancestry_proportions.tsv", sep="\t")
fam = pd.read_csv(ROOT / "student_family_metadata.tsv", sep="\t")
kin = pd.read_csv(RES / "kinship_pairs.tsv", sep="\t")

# --- UMAP on PC1-10 ---
N_PC = 10
X = pca[[f"PC{i+1}" for i in range(N_PC)]].values
reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42, n_components=2)
emb = reducer.fit_transform(X)
pca["UMAP1"] = emb[:, 0]
pca["UMAP2"] = emb[:, 1]

SUPERPOP_COLORS = {"AFR": "#e41a1c", "EUR": "#377eb8", "EAS": "#4daf4a",
                   "SAS": "#984ea3", "AMR": "#ff7f00", "STUDENT": "#222"}
fig, ax = plt.subplots(figsize=(11, 9))
refs = pca[~pca["is_student"]]
studs = pca[pca["is_student"]]
for sp, sub in refs.groupby("superpopulation"):
    ax.scatter(sub["UMAP1"], sub["UMAP2"], s=40, alpha=0.65,
               c=SUPERPOP_COLORS.get(sp, "#999"), label=sp, edgecolors="none")
ax.scatter(studs["UMAP1"], studs["UMAP2"], s=110, marker="x",
           c="black", linewidths=1.6, label="students")
for _, r in studs.iterrows():
    ax.annotate(r["IID"], (r["UMAP1"], r["UMAP2"]), fontsize=6,
                xytext=(3, 3), textcoords="offset points", alpha=0.85)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("UMAP of first 10 PCs (1000G refs + students)")
ax.legend(loc="best", fontsize=10, framealpha=0.95)
fig.tight_layout()
fig.savefig(RES / "umap.png", dpi=140)
plt.close(fig)
print("Saved umap.png")

# --- Build final per-student summary table ---
summary = fam[["sample_id", "age"]].copy()
summary = summary.merge(pred[["sample_id", "pred_superpop", "pred_superpop_name",
                              "pred_superpop_prob", "pred_pop1", "pred_pop1_name",
                              "pred_pop1_prob", "pred_pop2", "pred_pop2_prob",
                              "pred_pop3", "pred_pop3_prob"]], on="sample_id")
summary = summary.merge(anc, on="sample_id")
summary.rename(columns={"AFR": "anc_AFR", "AMR": "anc_AMR", "EAS": "anc_EAS",
                        "EUR": "anc_EUR", "SAS": "anc_SAS"}, inplace=True)

# Assign family id
import networkx as nx
G = nx.Graph()
students = set(fam["sample_id"])
G.add_nodes_from(students)
for _, r in kin.iterrows():
    if r["KINSHIP"] >= 0.0884:  # 1st/2nd degree only for family-finding
        G.add_edge(r["ID1"], r["ID2"])
fam_id = {}
for i, comp in enumerate(nx.connected_components(G)):
    for n in comp:
        fam_id[n] = i + 1 if len(comp) > 1 else 0
summary["family_id"] = summary["sample_id"].map(fam_id).fillna(0).astype(int)

deg = dict(G.degree())
summary["n_close_relatives"] = summary["sample_id"].map(deg).fillna(0).astype(int)

# Reorder + format
cols = ["sample_id", "age", "family_id", "n_close_relatives",
        "pred_superpop", "pred_superpop_prob",
        "pred_pop1", "pred_pop1_name", "pred_pop1_prob",
        "pred_pop2", "pred_pop2_prob",
        "pred_pop3", "pred_pop3_prob",
        "anc_AFR", "anc_AMR", "anc_EAS", "anc_EUR", "anc_SAS"]
summary = summary[cols]
summary.to_csv(RES / "student_summary.tsv", sep="\t", index=False, float_format="%.3f")
print(f"Saved {RES/'student_summary.tsv'}")
print("\nFinal summary (first 15):")
print(summary.head(15).to_string(index=False))

# Counts
print("\n--- Counts ---")
print("Predicted superpopulations:")
print(summary["pred_superpop"].value_counts())
print("\nPredicted populations (top):")
print(summary["pred_pop1"].value_counts())
print(f"\nTotal families: {summary['family_id'].max()}")
print(f"Students with no close kin in dataset: {(summary['n_close_relatives']==0).sum()}")
