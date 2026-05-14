#!/usr/bin/env bash
# Digital Genealogist — full pipeline runner.
#
# Inputs (must be present in repo root):
#   families_plus_popref.vcf            — multi-sample VCF (NOT in git, ~14 GB)
#   student_reference_metadata.tsv      — 1000G reference labels (sample_id, population, superpopulation, ...)
#   student_family_metadata.tsv         — student labels (sample_id, age)
#   tools/plink2[.exe]                  — install with ./setup_tools.sh
#
# Output:
#   work/      — PLINK 2 intermediates (gitignored)
#   results/   — TSV tables, PNG figures, dashboard.html, REPORT.md
#   site/      — Netlify-ready static site (single index.html)

set -euo pipefail

# -- platform-dependent plink2 binary --
if command -v plink2 >/dev/null 2>&1; then
  PLINK=plink2
elif [[ -x ./tools/plink2 ]]; then
  PLINK=./tools/plink2
elif [[ -x ./tools/plink2.exe ]]; then
  PLINK=./tools/plink2.exe
else
  echo "[!] plink2 not found. Run ./setup_tools.sh first."
  exit 1
fi
echo "Using plink2: $PLINK"

# Required inputs
for f in families_plus_popref.vcf student_reference_metadata.tsv student_family_metadata.tsv; do
  if [[ ! -f "$f" ]]; then
    echo "[!] Missing required input: $f"
    exit 1
  fi
done

mkdir -p work results

# -- 1. QC + VCF → pfile (binary) ----------------------------------------
echo "[1/5] QC + VCF → pfile"
"$PLINK" \
  --vcf families_plus_popref.vcf \
  --max-alleles 2 --snps-only --autosome \
  --maf 0.05 --geno 0.05 --hwe 1e-10 \
  --set-all-var-ids '@:#:$r:$a' \
  --new-id-max-allele-len 50 missing \
  --make-pgen --threads 4 --memory 8000 \
  --out work/data_qc

# -- 2. LD prune ---------------------------------------------------------
echo "[2/5] LD-prune"
"$PLINK" --pfile work/data_qc --indep-pairwise 200 50 0.2 \
  --threads 4 --memory 8000 --out work/prune
"$PLINK" --pfile work/data_qc --extract work/prune.prune.in --make-pgen \
  --threads 4 --memory 8000 --out work/data_pruned

# -- 3. PCA + KING -------------------------------------------------------
echo "[3/5] PCA + KING kinship"
"$PLINK" --pfile work/data_pruned --pca 20 \
  --threads 4 --memory 8000 --out work/pca_all
"$PLINK" --pfile work/data_pruned --make-king-table --king-table-filter 0.0442 \
  --threads 4 --memory 8000 --out work/king

# Build per-superpop sample-keep files
python - <<'PY'
import pandas as pd
ref = pd.read_csv('student_reference_metadata.tsv', sep='\t')
for s, sub in ref.groupby('superpopulation'):
    with open(f'work/keep_{s}.txt','w') as f:
        f.write('#IID\n')
        for sid in sub['sample_id']:
            f.write(f'{sid}\n')

fam = pd.read_csv('student_family_metadata.tsv', sep='\t')
with open('work/keep_students.txt','w') as f:
    f.write('#IID\n')
    for sid in fam['sample_id']:
        f.write(f'{sid}\n')
PY

for S in AFR AMR EAS EUR SAS; do
  "$PLINK" --pfile work/data_pruned --keep work/keep_${S}.txt --freq \
    --threads 4 --memory 4000 --out work/freq_${S}
done
"$PLINK" --pfile work/data_pruned --keep work/keep_students.txt --export A \
  --threads 4 --memory 4000 --out work/stud_raw

# -- 4. Python analysis --------------------------------------------------
echo "[4/5] Python analysis"
python scripts/01_pca_and_classify.py
python scripts/02_kinship_families.py
python scripts/03_ancestry_nnls.py
python scripts/05_cascade_classify.py   # internally runs 04 (UMAP + summary)

# -- 5. Visualisations + site -------------------------------------------
echo "[5/5] Visualisations + Netlify site"
python scripts/06_dashboard.py
python scripts/07_pedigree_pies.py
python scripts/08_radial_views.py
python scripts/09_build_site.py

echo
echo "Done."
echo "  Tables / figures:    results/"
echo "  Static site:         site/  (drag onto https://app.netlify.com/drop)"
echo "  Interactive report:  results/dashboard.html"
