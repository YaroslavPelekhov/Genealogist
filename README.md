# Digital Genealogist

Population structure, ancestry composition and kinship reconstruction for 70
anonymous students from a 14 GB multi-sample VCF (5.9 M variants, 278 samples,
GRCh38), with a 1000 Genomes 26-population reference panel.

[![Netlify Status](https://api.netlify.com/api/v1/badges/REPLACE_WITH_SITE_ID/deploy-status)](https://app.netlify.com/sites/glowing-raindrop-d5bac1/deploys)
[![Live demo](https://img.shields.io/badge/live%20demo-glowing--raindrop--d5bac1.netlify.app-7cd0ff)](https://glowing-raindrop-d5bac1.netlify.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![plink2](https://img.shields.io/badge/plink2-required-orange)
![Plotly](https://img.shields.io/badge/plotly-3.4-9cf)

> **Live dashboard:** <https://glowing-raindrop-d5bac1.netlify.app/>

> Reproducible pipeline: PLINK 2 (QC, LD-prune, PCA, KING) + Python
> (scikit-learn, scipy NNLS, networkx, umap-learn, plotly).

---

## Headline findings

| | |
|---|---|
| **70 students** | divided into **15 families** through 1st/2nd-degree kinship |
| **127 related pairs** | 68 parent–child · 20 full-sibling · 36 second-degree · **3 MZ-twin pairs** (s92/s93, s16/s94, s67/s95 — all confirmed by matching ages) |
| **Superpopulation classification** | RF on PC1–10, **99.0 %** 5-fold CV |
| **Population classification (26 1000G pops)** | cascade RF → kNN, **65.4 %** 5-fold CV |
| **Global ancestry** | supervised NNLS on 181 k LD-pruned SNPs, 5-way AFR/AMR/EAS/EUR/SAS |

## Visual outputs

| | |
|---|---|
| **Interactive HTML dashboard** | [`results/dashboard.html`](results/dashboard.html) — 3D PCA, 2D PCA, UMAP, ancestry barplot, sunburst, kinship network |
| **Radial "family crest"** | [`results/radial_families.png`](results/radial_families.png) — 70 students on a circle, grouped by family, with intra-family Bezier kinship arcs |
| **Pedigree trees with ancestry pies** | [`results/family_pedigrees.png`](results/family_pedigrees.png) — Y-axis = age, each node = 5-way ancestry pie, edges by relation type |
| **Netlify-ready single-page site** | [`site/`](site/) — drag-and-drop publishable |

## Pipeline

```
families_plus_popref.vcf  (14 GB, 5 906 144 variants, 278 samples)
        │
        │  plink2 QC: bi-allelic SNPs, autosomes, MAF≥0.05, geno≤0.05, HWE>1e-10
        ▼
work/data_qc.{pgen,pvar,psam}                     (4 993 075 variants)
        │
        │  plink2 --indep-pairwise 200 50 0.2
        ▼
work/data_pruned.{pgen,pvar,psam}                   (181 101 SNPs)
        │
        ├─► plink2 --pca 20         → work/pca_all.eigenvec / .eigenval
        │
        ├─► plink2 --make-king-table → work/king.kin0  (127 related pairs)
        │
        ├─► plink2 --freq (×5 superpops) + --export A (students)
        │
        ▼
Python:  PCA visualisations, UMAP, RF+kNN cascade classifier,
         NNLS ancestry, kinship → family graphs, pedigrees, dashboard
        │
        ▼
results/ + site/
```

## Quick start

### 0. Prerequisites

- **Python 3.10+** (any platform)
- **plink2** binary
- **~70 GB** of free disk space during the run (intermediates)
- **~16 GB RAM**

### 1. Clone

```bash
git clone <your-fork-url> digital-genealogist
cd digital-genealogist
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
./setup_tools.sh             # downloads plink2 into ./tools/
```

### 3. Provide inputs

Place these files in the repo root (not in git — see `.gitignore`):

| File | Description |
|---|---|
| `families_plus_popref.vcf` | The multi-sample VCF (~14 GB) |
| `student_reference_metadata.tsv` | Cols: `sample_id`, `population`, `superpopulation`, `population_name`, `superpopulation_name` |
| `student_family_metadata.tsv` | Cols: `sample_id`, `age` |

### 4. Run

```bash
./run_pipeline.sh
```

Expected runtime on a 4-core workstation: **~15 min** (VCF→pfile is the bottleneck).
Output:
- `results/` — TSV tables + PNGs + interactive dashboard + REPORT.md
- `site/` — Netlify-ready static site

### 5. Deploy the site (optional)

**Drag-and-drop:** drop `site/` onto <https://app.netlify.com/drop>.

**Or Netlify CLI:**
```bash
npm install -g netlify-cli
cd site && netlify deploy --prod
```

To refresh the deployed site after re-running the pipeline, just drop `site/`
onto Netlify Drop again (or `netlify deploy --prod` from `site/`).

**Getting the Netlify badge:** find your **Site API ID** in
*Site settings → General → Site information → API ID* (a UUID like
`1a2b3c4d-…`) and replace `REPLACE_WITH_SITE_ID` in the badge URLs at the top
of this README.

## Repository layout

```
.
├── README.md                      <-- this file
├── LICENSE                        MIT
├── requirements.txt               python deps
├── setup_tools.sh                 install plink2
├── run_pipeline.sh                end-to-end runner
│
├── scripts/                       9 analysis scripts (see below)
├── results/                       generated TSVs + PNGs + REPORT.md + dashboard.html
└── site/                          Netlify-ready single-page site
    ├── index.html
    ├── netlify.toml
    ├── assets/
    └── data/

# NOT in git:
# families_plus_popref.vcf  (~14 GB)
# work/                     (~7 GB PLINK intermediates)
# tools/plink2[.exe]        (downloaded by setup_tools.sh)
```

## Scripts

| # | Script | Role |
|---|---|---|
| 01 | `01_pca_and_classify.py` | PCA + flat RF classifier (superseded by 05) |
| 02 | `02_kinship_families.py` | KING-table parsing, family connected components, basic family-tree PNG |
| 03 | `03_ancestry_nnls.py` | Supervised NNLS ancestry on superpop allele frequencies |
| 04 | `04_umap_and_summary.py` | UMAP of PCs + final `student_summary.tsv` |
| 05 | `05_cascade_classify.py` | Cascade RF→kNN classifier; internally calls 04 |
| 06 | `06_dashboard.py` | Interactive Plotly dashboard (`dashboard.html`) |
| 07 | `07_pedigree_pies.py` | Pedigree-style family trees with ancestry pie nodes |
| 08 | `08_radial_views.py` | Radial "family crest", Sankey, clustered ancestry |
| 09 | `09_build_site.py` | Netlify static-site assembly |

## Method notes

- **QC**: bi-allelic autosomal SNPs only, MAF ≥ 0.05, missingness ≤ 5 %, HWE > 1e-10.
  These thresholds are loose enough for the 1000G + admixed-student mix; tighten
  HWE if all samples are from a single homogeneous population.
- **LD pruning**: `--indep-pairwise 200 50 0.2` → ~181 k SNPs from ~5 M.
- **PCA**: joint PCA on all 278 samples. With 70 / 278 ratio, students do not
  dominate the PCs; PC1 27.7 %, PC2 16.4 % of variance.
- **Classification**: cascade — RandomForest predicts superpopulation
  (5 classes), then per-superpopulation kNN-5 predicts the population
  (26 classes). Cross-validated: 99.0 % / 65.4 %.
- **Kinship**: KING-robust kinship (Manichaikul et al. 2010). Thresholds:
  >0.354 MZ/dup · 0.177–0.354 1st-degree · 0.0884–0.177 2nd · 0.0442–0.0884 3rd.
  Parent–child distinguished from full-siblings by IBS0 ≈ 0 (must share an
  allele at every locus).
- **Ancestry**: supervised NNLS — for each student, solve
  `argmin_α Σ_j (g_ij − 2 Σ_k α_k p_kj)²`, α ≥ 0, Σα = 1, on 181 k SNPs.
  Caveat: 1000G AMR populations (MXL/PEL/CLM/PUR) are themselves admixed; this
  produces a small spurious AMR component in every non-AMR student. For
  strictly continental ancestry, use HGDP/SGDP references or run an
  unsupervised ADMIXTURE with K=5.

## Limitations / future work

- **Local ancestry / chromosome painting** — needs phasing (SHAPEIT) + RFMix/Gnomix
- **Unsupervised ADMIXTURE K=5** — would remove the AMR-reference bias above
- **Runs of Homozygosity (ROH)** + F coefficient — detect inbreeding
- **Direction reconstruction** (parent → child) by age within IBS0-confirmed pairs

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If this pipeline is useful in academic work, please cite the underlying tools:

- Chang CC, et al. *Second-generation PLINK: rising to the challenge of larger
  and richer datasets.* GigaScience 2015; 4:7.
- Manichaikul A, et al. *Robust relationship inference in genome-wide
  association studies.* Bioinformatics 2010; 26(22):2867–2873.
- McInnes L, Healy J, Melville J. *UMAP: Uniform Manifold Approximation and
  Projection for Dimension Reduction.* arXiv:1802.03426.
- 1000 Genomes Project Consortium. *A global reference for human genetic
  variation.* Nature 2015; 526:68–74.
