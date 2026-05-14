"""Solution B — GRM-based pipeline.
Step 1: Eigendecompose the ref-only GRM → reference PC axes,
        project students onto these axes (LASER / FastPCA-style).

Background:
  GRM_{ij} = (1/M) Σ_k (g_ik − 2p_k)(g_jk − 2p_k) / [2 p_k (1−p_k)]
  Top eigenvectors of GRM == PCA scores (Patterson, Price & Reich 2006).

  For reference projection of a new sample s onto reference PC axis k:
    PC_k(s) = (1/√λ_k) * Σ_{i ∈ ref} v_{ki} * GRM(s, i)
  where (λ_k, v_k) is the k-th eigenpair of the *ref-only* GRM.

  This isolates students from influencing the PC axes (a problem with joint
  PCA when the new cohort is large or admixed).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
WORK = ROOT / "work"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

def load_rel(prefix):
    p = Path(prefix)
    with open(p.with_suffix(".rel.id")) as f:
        lines = [l.strip().split("\t") for l in f if l.strip() and not l.strip().startswith("#")]
    ids = [row[-1] for row in lines]    # take IID (last col if 2-col, else only col)
    n = len(ids)
    G = np.zeros((n, n))
    with open(p.with_suffix(".rel")) as f:
        for i, line in enumerate(f):
            vals = [float(x) for x in line.split()]
            G[i, :i+1] = vals
            G[:i+1, i] = vals
    return G, ids

# 1. Load ref-only and joint GRMs
G_ref, ids_ref = load_rel(WORK / "grm_refs_plain")
G_joint, ids_joint = load_rel(WORK / "grm_joint")
print(f"ref GRM: {G_ref.shape}")
print(f"joint GRM: {G_joint.shape}")

# Index map for joint
ix_ref = np.array([ids_joint.index(i) for i in ids_ref])
ref_set = set(ids_ref)
student_ids = [i for i in ids_joint if i not in ref_set]
ix_stud = np.array([ids_joint.index(i) for i in student_ids])
print(f"ref n={len(ids_ref)}, student n={len(student_ids)}")

# 2. Sanity: ref-only GRM should match joint[ix_ref, ix_ref]
sub = G_joint[np.ix_(ix_ref, ix_ref)]
delta = np.abs(sub - G_ref)
print(f"joint[ref,ref] vs ref-GRM:  max|diff| = {delta.max():.4f},  mean|diff| = {delta.mean():.4f}")
# (small differences expected: allele-freq estimates change when students are included)

# 3. Eigendecompose ref-only GRM
N_REF = G_ref.shape[0]
# GRM is positive-semidefinite; use eigh for symmetric
eigvals, eigvecs = np.linalg.eigh(G_ref)
# sort descending
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]
N_PC = 20
top_lambda = eigvals[:N_PC]
top_V = eigvecs[:, :N_PC]
total_var = eigvals.clip(min=0).sum()
var_expl = top_lambda / total_var
print(f"\nTop 10 eigenvalues: {top_lambda[:10].round(2)}")
print(f"Variance explained (top 10): {var_expl[:10].round(3)}")

# Reference PC scores: PC_k(i) = sqrt(λ_k) * v_{ki}  (standard convention so cov(PC)=λ)
ref_pcs = top_V * np.sqrt(top_lambda)[None, :]
ref_pc_df = pd.DataFrame(ref_pcs, columns=[f"PC{k+1}" for k in range(N_PC)])
ref_pc_df.insert(0, "IID", ids_ref)
ref_pc_df["is_student"] = False

# 4. Project students: PC_k(s) = (1/√λ_k) * Σ_i v_{ki} * G_joint[s, i]
#    For multiple students s, this is the matrix:
#    PC_stud (n_stud × N_PC) = G_joint[stud, ref] @ V * diag(1/√λ)
G_sr = G_joint[np.ix_(ix_stud, ix_ref)]  # n_stud × n_ref
stud_pcs = G_sr @ top_V * (1.0 / np.sqrt(top_lambda))[None, :]
stud_pc_df = pd.DataFrame(stud_pcs, columns=[f"PC{k+1}" for k in range(N_PC)])
stud_pc_df.insert(0, "IID", student_ids)
stud_pc_df["is_student"] = True

projected = pd.concat([ref_pc_df, stud_pc_df], ignore_index=True)
projected.to_csv(RES / "grm_projected_pca.tsv", sep="\t", index=False, float_format="%.6f")
print(f"\nSaved {RES/'grm_projected_pca.tsv'}  shape={projected.shape}")

# Also save eigenvalues
pd.DataFrame({"PC": np.arange(1, N_PC+1),
              "eigenvalue": top_lambda,
              "variance_fraction": var_expl}).to_csv(
    RES / "grm_eigenvalues.tsv", sep="\t", index=False, float_format="%.6g")
print(f"Saved {RES/'grm_eigenvalues.tsv'}")

# 5. Print first 5 students' projected coordinates
print("\nFirst 5 students, top-5 projected PCs:")
print(stud_pc_df.head().to_string(index=False))
