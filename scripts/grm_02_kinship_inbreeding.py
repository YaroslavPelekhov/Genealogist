"""Solution B — GRM-based kinship + inbreeding analysis.

From the joint GRM:
  - Diagonal: G_ii = 1 + F_i (per-sample inbreeding coefficient)
  - Off-diagonal: G_ij ≈ 2 * kinship_ij  (VanRaden 2008 GRM ~ 2 × IBD kinship)

We:
  1. Compute F_i for every sample (diagonal − 1).
  2. Threshold off-diagonal GRM values to derive related pairs;
     compare with KING-robust kinship from Solution A.
  3. Note: GRM is NOT robust to population stratification — within-
     population pairs have inflated GRM even if unrelated. So GRM-kinship
     is reliable only when both samples are from the same (homogeneous)
     population, or after PC-based correction.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
WORK = ROOT / "work"
RES = ROOT / "results"

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

G_joint, ids_joint = load_rel(WORK / "grm_joint")
n = len(ids_joint)

# --- 1. Inbreeding F = diag − 1 ---
F = np.diag(G_joint) - 1.0
ref_meta = pd.read_csv(ROOT / "student_reference_metadata.tsv", sep="\t")
ref_meta_map = ref_meta.set_index("sample_id")
fam_meta = pd.read_csv(ROOT / "student_family_metadata.tsv", sep="\t").set_index("sample_id")

F_df = pd.DataFrame({"sample_id": ids_joint, "F_inbreeding": F})
F_df["is_student"] = ~F_df["sample_id"].str.startswith("ref")
F_df["population"] = F_df["sample_id"].map(ref_meta_map["population"]).fillna("STUDENT")
F_df["superpopulation"] = F_df["sample_id"].map(ref_meta_map["superpopulation"]).fillna("STUDENT")
F_df["age"] = F_df["sample_id"].map(fam_meta["age"])

F_df.to_csv(RES / "grm_F_inbreeding.tsv", sep="\t", index=False, float_format="%.4f")
print(f"F_inbreeding (n={n}):")
print(f"  range: [{F.min():.3f}, {F.max():.3f}], mean={F.mean():.3f}")
print(f"\nF by group:")
print(F_df.groupby("superpopulation")["F_inbreeding"].agg(["count","mean","min","max"]).round(3))

# Top inbred individuals
print("\nTop 10 |F| (potentially inbred or outliers):")
top = F_df.reindex(F_df["F_inbreeding"].abs().sort_values(ascending=False).index).head(10)
print(top[["sample_id","is_student","superpopulation","age","F_inbreeding"]].to_string(index=False))

# --- 2. Off-diagonal GRM → related pairs ---
iu = np.triu_indices(n, k=1)
pair_grm = G_joint[iu]
# GRM threshold: G > 0.354 = MZ-twin; 0.177–0.354 = 1st degree; 0.088–0.177 = 2nd, etc.
# Same numerical thresholds as KING kinship, since GRM ≈ 2*kinship in expectation.
def cls(g):
    if g > 0.354: return "MZ/dup"
    if g > 0.177: return "1st-degree"
    if g > 0.0884: return "2nd-degree"
    if g > 0.0442: return "3rd-degree"
    return "unrelated"

pairs = []
for k, (i, j) in enumerate(zip(*iu)):
    g = pair_grm[k]
    if g > 0.0442:  # 3rd degree or closer
        pairs.append({"ID1": ids_joint[i], "ID2": ids_joint[j],
                      "GRM_value": g, "GRM_class": cls(g)})
grm_kin = pd.DataFrame(pairs)
print(f"\nGRM-implied related pairs (G > 0.0442): {len(grm_kin)}")
print(grm_kin["GRM_class"].value_counts())

# Within-students only
grm_kin["both_students"] = grm_kin["ID1"].str.startswith("s") & grm_kin["ID2"].str.startswith("s") \
                           & ~grm_kin["ID1"].str.startswith("ref") & ~grm_kin["ID2"].str.startswith("ref")
ss = grm_kin[grm_kin["both_students"]]
print(f"  student-student pairs: {len(ss)}")
print(f"  ref-ref pairs:         {(~grm_kin['both_students'] & ~(grm_kin['ID1'].str.startswith('s') ^ grm_kin['ID2'].str.startswith('s'))).sum()}")
ref_ref = grm_kin[(grm_kin['ID1'].str.startswith('ref')) & (grm_kin['ID2'].str.startswith('ref'))]
mixed = grm_kin[grm_kin['ID1'].str.startswith('ref') != grm_kin['ID2'].str.startswith('ref')]
print(f"  ref-ref: {len(ref_ref)}, student-ref: {len(mixed)}")

grm_kin.drop(columns=["both_students"]).to_csv(
    RES / "grm_related_pairs.tsv", sep="\t", index=False, float_format="%.4f")
print(f"\nSaved {RES/'grm_related_pairs.tsv'}")

# --- 3. Cross-check with KING ---
king = pd.read_csv(RES / "kinship_pairs.tsv", sep="\t")
print(f"\nKING related pairs: {len(king)}")
king_pairs = set(frozenset([r['ID1'], r['ID2']]) for _, r in king.iterrows())
grm_pairs = set(frozenset([r['ID1'], r['ID2']]) for _, r in ss.iterrows())
common = king_pairs & grm_pairs
print(f"  KING ∩ GRM (student pairs): {len(common)} / KING={len(king_pairs)}, GRM={len(grm_pairs)}")
king_only = king_pairs - grm_pairs
grm_only = grm_pairs - king_pairs
print(f"  KING-only: {len(king_only)},  GRM-only: {len(grm_only)}")
if king_only:
    print("  Examples of KING-only (likely cross-population pairs where GRM under-calls):")
    for p in list(king_only)[:5]:
        a, b = sorted(p)
        kv = king[((king['ID1']==a)&(king['ID2']==b))|((king['ID1']==b)&(king['ID2']==a))]['KINSHIP'].values[0]
        i_a = ids_joint.index(a); i_b = ids_joint.index(b)
        gv = G_joint[i_a, i_b]
        print(f"    {a}-{b}: KING={kv:.3f}, GRM={gv:.3f}")
if grm_only:
    print(f"  Examples of GRM-only (likely population-structure false positives):")
    for p in list(grm_only)[:5]:
        a, b = sorted(p)
        i_a = ids_joint.index(a); i_b = ids_joint.index(b)
        gv = G_joint[i_a, i_b]
        print(f"    {a}-{b}: GRM={gv:.3f}, KING=<0.0442 (filtered)")

# --- 4. Pairwise comparison plot data ---
# For all student-student pairs (both methods on same set), match values
king_lookup = {frozenset([r['ID1'], r['ID2']]): r['KINSHIP'] for _, r in king.iterrows()}
stud_ids = [i for i in ids_joint if not i.startswith("ref")]
stud_idx = {s: ids_joint.index(s) for s in stud_ids}
cmp_rows = []
for i in range(len(stud_ids)):
    for j in range(i+1, len(stud_ids)):
        a, b = stud_ids[i], stud_ids[j]
        g = G_joint[stud_idx[a], stud_idx[b]]
        k = king_lookup.get(frozenset([a, b]), 0.0)  # 0 if filtered out
        cmp_rows.append({"ID1": a, "ID2": b, "GRM": g, "KING": k})
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(RES / "grm_vs_king_pairs.tsv", sep="\t", index=False, float_format="%.4f")
print(f"\nSaved comparison: {RES/'grm_vs_king_pairs.tsv'}  ({len(cmp_df)} student-student pairs)")
r = np.corrcoef(cmp_df['GRM'], cmp_df['KING'])[0,1]
print(f"GRM vs KING Pearson r over all 70 students: {r:.4f}")
