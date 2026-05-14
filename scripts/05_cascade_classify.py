"""Cascade classifier: first predict superpopulation, then population within superpop.
Re-export predictions and update student_summary.tsv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
WORK = ROOT / "work"
RES = ROOT / "results"

pca = pd.read_csv(RES / "pca_coords.tsv", sep="\t")
ref_meta = pd.read_csv(ROOT / "student_reference_metadata.tsv", sep="\t")

N_PCS = 10
PC_COLS = [f"PC{i+1}" for i in range(N_PCS)]

X_ref = pca.loc[~pca["is_student"], PC_COLS].values
y_super = pca.loc[~pca["is_student"], "superpopulation"].values
y_pop = pca.loc[~pca["is_student"], "population"].values
ref_ids = pca.loc[~pca["is_student"], "IID"].values

X_stud = pca.loc[pca["is_student"], PC_COLS].values
stud_ids = pca.loc[pca["is_student"], "IID"].values

# --- Stage 1: superpop classifier ---
clf_super = RandomForestClassifier(n_estimators=600, random_state=0, n_jobs=-1)
clf_super.fit(X_ref, y_super)
super_classes = clf_super.classes_
prob_super = clf_super.predict_proba(X_stud)

# --- Stage 2: per-superpop population classifier ---
# Train one kNN per superpop using its reference samples
pop_clf_per_super = {}
for sp in super_classes:
    mask = (y_super == sp)
    pop_clf_per_super[sp] = KNeighborsClassifier(n_neighbors=5).fit(X_ref[mask], y_pop[mask])

# Marginal P(pop | sample) = sum_sp P(sp|x) * P(pop|x, sp)
all_pops = sorted(set(y_pop))
pop_idx = {p: i for i, p in enumerate(all_pops)}
prob_pop_all = np.zeros((len(stud_ids), len(all_pops)))
for sp_i, sp in enumerate(super_classes):
    clf = pop_clf_per_super[sp]
    p_pop_given_sp = clf.predict_proba(X_stud)
    for j, pname in enumerate(clf.classes_):
        prob_pop_all[:, pop_idx[pname]] += prob_super[:, sp_i] * p_pop_given_sp[:, j]
# Normalize (already sums to 1 because P(sp) sums to 1 and per-sp sums to 1)

pop_to_name = ref_meta.set_index("population")["population_name"].to_dict()
super_to_name = ref_meta.set_index("superpopulation")["superpopulation_name"].to_dict()

pred_super = super_classes[prob_super.argmax(axis=1)]
rows = []
for i, sid in enumerate(stud_ids):
    order = np.argsort(prob_pop_all[i])[::-1][:3]
    rows.append({
        "sample_id": sid,
        "pred_superpop": pred_super[i],
        "pred_superpop_name": super_to_name.get(pred_super[i], ""),
        "pred_superpop_prob": float(prob_super[i].max()),
        "pred_pop1": all_pops[order[0]],
        "pred_pop1_name": pop_to_name.get(all_pops[order[0]], ""),
        "pred_pop1_prob": float(prob_pop_all[i, order[0]]),
        "pred_pop2": all_pops[order[1]],
        "pred_pop2_prob": float(prob_pop_all[i, order[1]]),
        "pred_pop3": all_pops[order[2]],
        "pred_pop3_prob": float(prob_pop_all[i, order[2]]),
    })
out = pd.DataFrame(rows)
out.to_csv(RES / "population_predictions.tsv", sep="\t", index=False, float_format="%.3f")
print("Updated population_predictions.tsv with cascade classifier:")
print(out.head(20).to_string(index=False))

# CV: how well does the cascade do?
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
correct_super = 0
correct_pop = 0
n = 0
for tr, te in skf.split(X_ref, y_pop):
    cs = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(X_ref[tr], y_super[tr])
    ps = cs.predict(X_ref[te])
    correct_super += (ps == y_super[te]).sum()
    # population predicted via cascade
    super_classes_cv = cs.classes_
    pp_per_sp = {sp: KNeighborsClassifier(5).fit(X_ref[tr][y_super[tr]==sp], y_pop[tr][y_super[tr]==sp])
                 for sp in super_classes_cv}
    pred_super_cv = cs.predict(X_ref[te])
    pred_pop_cv = []
    for k, x in enumerate(X_ref[te]):
        pred_pop_cv.append(pp_per_sp[pred_super_cv[k]].predict(x.reshape(1, -1))[0])
    pred_pop_cv = np.array(pred_pop_cv)
    correct_pop += (pred_pop_cv == y_pop[te]).sum()
    n += len(te)
print(f"\nCascade CV: superpop {correct_super/n:.3f}, population {correct_pop/n:.3f}")

# Re-run summary with updated predictions
import subprocess
subprocess.run(["python", str(ROOT/"scripts/04_umap_and_summary.py")], check=True)
