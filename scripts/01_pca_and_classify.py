"""PCA visualization + population classification of student samples.

Inputs:
  work/pca_all.eigenvec, work/pca_all.eigenval
  student_reference_metadata.tsv, student_family_metadata.tsv
Outputs:
  results/pca_superpop.png, results/pca_population.png, results/pca_zoom_students.png
  results/scree.png
  results/population_predictions.tsv
  results/superpop_predictions.tsv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
WORK = ROOT / "work"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# --- Load PCA ---
eig = pd.read_csv(WORK / "pca_all.eigenvec", sep="\t")
# plink2 eigenvec: #IID PC1..PCk (no FID column when --double-id not used and no FIDs)
eig = eig.rename(columns={"#IID": "IID", "#FID": "FID"})
if "IID" not in eig.columns:
    eig["IID"] = eig.iloc[:, 0]
pc_cols = [c for c in eig.columns if c.startswith("PC")]
print(f"Loaded PCA: {len(eig)} samples, {len(pc_cols)} PCs")

eigval = pd.read_csv(WORK / "pca_all.eigenval", header=None, names=["eigval"])
var_explained = eigval["eigval"] / eigval["eigval"].sum()
print("Variance explained (top 10):", var_explained.head(10).round(3).tolist())

# --- Load metadata ---
ref = pd.read_csv(ROOT / "student_reference_metadata.tsv", sep="\t")
fam = pd.read_csv(ROOT / "student_family_metadata.tsv", sep="\t")

# Merge
ref_lab = ref[["sample_id", "population", "superpopulation", "population_name", "superpopulation_name"]].copy()
fam_lab = fam[["sample_id", "age"]].copy()
fam_lab["population"] = "STUDENT"
fam_lab["superpopulation"] = "STUDENT"

df = eig.merge(ref_lab, left_on="IID", right_on="sample_id", how="left")
df = df.merge(fam_lab[["sample_id", "age"]], left_on="IID", right_on="sample_id", how="left", suffixes=("", "_fam"))
df["is_student"] = df["IID"].isin(fam["sample_id"])
df["population"] = df["population"].fillna("STUDENT")
df["superpopulation"] = df["superpopulation"].fillna("STUDENT")
print("Samples by type:", df["is_student"].value_counts().to_dict())

# --- Scree plot ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, len(var_explained) + 1), var_explained * 100, color="steelblue")
ax.set_xlabel("PC")
ax.set_ylabel("% variance explained")
ax.set_title("PCA scree plot")
fig.tight_layout()
fig.savefig(RES / "scree.png", dpi=130)
plt.close(fig)

# --- PCA scatter: superpopulation ---
SUPERPOP_COLORS = {
    "AFR": "#e41a1c", "EUR": "#377eb8", "EAS": "#4daf4a",
    "SAS": "#984ea3", "AMR": "#ff7f00", "STUDENT": "#222222",
}

def scatter(ax, df, col, color_map, pc_x="PC1", pc_y="PC2", student_size=70, ref_size=28):
    refs = df[~df["is_student"]]
    studs = df[df["is_student"]]
    for label, sub in refs.groupby(col):
        ax.scatter(sub[pc_x], sub[pc_y], s=ref_size, alpha=0.6,
                   c=color_map.get(label, "#999"), label=label, edgecolors="none")
    ax.scatter(studs[pc_x], studs[pc_y], s=student_size, marker="x",
               c="black", linewidths=1.6, label="students")
    ax.set_xlabel(f"{pc_x} ({var_explained[int(pc_x[2:])-1]*100:.1f}%)")
    ax.set_ylabel(f"{pc_y} ({var_explained[int(pc_y[2:])-1]*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
scatter(axes[0], df, "superpopulation", SUPERPOP_COLORS, "PC1", "PC2")
axes[0].legend(loc="best", fontsize=9, framealpha=0.9)
axes[0].set_title("PCA — 1000G superpopulations + students (PC1/PC2)")
scatter(axes[1], df, "superpopulation", SUPERPOP_COLORS, "PC3", "PC4")
axes[1].legend(loc="best", fontsize=9, framealpha=0.9)
axes[1].set_title("PCA — superpopulations (PC3/PC4)")
fig.tight_layout()
fig.savefig(RES / "pca_superpop.png", dpi=140)
plt.close(fig)
print("Saved pca_superpop.png")

# --- PCA: population (26 pops, distinct colors) ---
pops = sorted(ref["population"].unique())
cmap = plt.get_cmap("tab20", 26)
POP_COLORS = {p: cmap(i) for i, p in enumerate(pops)}
POP_COLORS["STUDENT"] = "#222"
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
scatter(axes[0], df, "population", POP_COLORS, "PC1", "PC2")
axes[0].legend(loc="best", fontsize=7, ncol=2, framealpha=0.9)
axes[0].set_title("PCA — 26 populations + students")
scatter(axes[1], df, "population", POP_COLORS, "PC3", "PC4")
axes[1].set_title("PCA — populations (PC3/PC4)")
fig.tight_layout()
fig.savefig(RES / "pca_population.png", dpi=140)
plt.close(fig)
print("Saved pca_population.png")

# --- Classification ---
N_PCS = 10
X_ref = df.loc[~df["is_student"], [f"PC{i+1}" for i in range(N_PCS)]].values
y_ref_super = df.loc[~df["is_student"], "superpopulation"].values
y_ref_pop = df.loc[~df["is_student"], "population"].values

X_stud = df.loc[df["is_student"], [f"PC{i+1}" for i in range(N_PCS)]].values
ids_stud = df.loc[df["is_student"], "IID"].values

# Cross-validate a few classifiers on refs
def cv_eval(name, clf, X, y):
    sc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return f"{name}: {sc.mean():.3f} +- {sc.std():.3f}"

print("\n--- Cross-validation on reference panel ---")
print("Superpopulation (5):")
for name, clf in [("kNN-5", KNeighborsClassifier(5)),
                  ("kNN-3", KNeighborsClassifier(3)),
                  ("SVC-rbf", Pipeline([("s", StandardScaler()), ("c", SVC(probability=True, kernel="rbf"))])),
                  ("RF", RandomForestClassifier(n_estimators=400, random_state=0))]:
    print(" ", cv_eval(name, clf, X_ref, y_ref_super))
print("Population (26):")
for name, clf in [("kNN-3", KNeighborsClassifier(3)),
                  ("kNN-5", KNeighborsClassifier(5)),
                  ("SVC-rbf", Pipeline([("s", StandardScaler()), ("c", SVC(probability=True, kernel="rbf"))])),
                  ("RF", RandomForestClassifier(n_estimators=400, random_state=0))]:
    print(" ", cv_eval(name, clf, X_ref, y_ref_pop))

# Use RF for both (robust + gives probabilities)
clf_super = RandomForestClassifier(n_estimators=600, random_state=0)
clf_super.fit(X_ref, y_ref_super)
clf_pop = RandomForestClassifier(n_estimators=600, random_state=0)
clf_pop.fit(X_ref, y_ref_pop)

prob_super = clf_super.predict_proba(X_stud)
prob_pop = clf_pop.predict_proba(X_stud)

pred_super = clf_super.predict(X_stud)
pred_pop = clf_pop.predict(X_stud)

# Names lookup
pop_to_name = ref.set_index("population")["population_name"].to_dict()
super_to_name = ref.set_index("superpopulation")["superpopulation_name"].to_dict()

# Top-3 populations per student
top3 = []
classes_pop = clf_pop.classes_
for i, sid in enumerate(ids_stud):
    probs = prob_pop[i]
    order = np.argsort(probs)[::-1][:3]
    top3.append({
        "sample_id": sid,
        "pred_superpop": pred_super[i],
        "pred_superpop_name": super_to_name.get(pred_super[i], ""),
        "pred_superpop_prob": float(prob_super[i].max()),
        "pred_pop1": classes_pop[order[0]],
        "pred_pop1_name": pop_to_name.get(classes_pop[order[0]], ""),
        "pred_pop1_prob": float(probs[order[0]]),
        "pred_pop2": classes_pop[order[1]],
        "pred_pop2_prob": float(probs[order[1]]),
        "pred_pop3": classes_pop[order[2]],
        "pred_pop3_prob": float(probs[order[2]]),
    })
out_pop = pd.DataFrame(top3)
out_pop.to_csv(RES / "population_predictions.tsv", sep="\t", index=False, float_format="%.3f")

# Superpopulation probabilities full
classes_super = clf_super.classes_
super_df = pd.DataFrame(prob_super, columns=[f"P_{c}" for c in classes_super])
super_df.insert(0, "sample_id", ids_stud)
super_df["pred_superpop"] = pred_super
super_df.to_csv(RES / "superpop_predictions.tsv", sep="\t", index=False, float_format="%.3f")

print("\n--- Sample predictions (first 10) ---")
print(out_pop.head(10).to_string(index=False))
print(f"\nSaved {RES / 'population_predictions.tsv'}")
print(f"Saved {RES / 'superpop_predictions.tsv'}")

# --- Save PCA coords with all metadata ---
df.to_csv(RES / "pca_coords.tsv", sep="\t", index=False, float_format="%.4f")
print(f"Saved {RES / 'pca_coords.tsv'}")
