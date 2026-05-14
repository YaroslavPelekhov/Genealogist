"""Build a Solution-B section into site/ and a standalone solution_b.html page.
Adds interactive Plotly views (3D projected PCA, GRM heatmap, scatter comparison).
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
WORK = ROOT / "work"
RES = ROOT / "results"
SITE = ROOT / "site"
ASSETS = SITE / "assets"
DATA = SITE / "data"

# Copy new artifacts
for png in ["grm_joint_heatmap.png", "grm_projected_pca.png", "grm_vs_plink_pca.png",
            "grm_F_inbreeding.png", "grm_vs_king_scatter.png"]:
    src = RES / png
    if src.exists():
        shutil.copy(src, ASSETS / png)
for tsv in ["grm_projected_pca.tsv", "grm_F_inbreeding.tsv",
            "grm_related_pairs.tsv", "grm_vs_king_pairs.tsv", "grm_eigenvalues.tsv"]:
    src = RES / tsv
    if src.exists():
        shutil.copy(src, DATA / tsv)

# --- Build interactive 3D projected PCA ---
SUPERPOP_COLORS = {"AFR": "#e41a1c", "EUR": "#377eb8", "EAS": "#4daf4a",
                   "SAS": "#984ea3", "AMR": "#ff7f00", "STUDENT": "#222"}

ref_meta = pd.read_csv(ROOT / "student_reference_metadata.tsv", sep="\t").set_index("sample_id")
fam_meta = pd.read_csv(ROOT / "student_family_metadata.tsv", sep="\t").set_index("sample_id")
summary = pd.read_csv(RES / "student_summary.tsv", sep="\t").set_index("sample_id")

proj = pd.read_csv(RES / "grm_projected_pca.tsv", sep="\t")
proj["superpopulation"] = proj["IID"].map(
    lambda i: ref_meta.loc[i, "superpopulation"] if i in ref_meta.index else "STUDENT")
proj["population"] = proj["IID"].map(
    lambda i: ref_meta.loc[i, "population"] if i in ref_meta.index else "STUDENT")
proj["population_name"] = proj["IID"].map(
    lambda i: ref_meta.loc[i, "population_name"] if i in ref_meta.index else "")

def hover(row):
    if row["is_student"]:
        age = fam_meta.loc[row["IID"], "age"] if row["IID"] in fam_meta.index else "?"
        try:
            s = summary.loc[row["IID"]]
            pop_pred = s["pred_pop1"]
            sp_pred = s["pred_superpop"]
        except KeyError:
            pop_pred, sp_pred = "?", "?"
        return (f"<b>{row['IID']}</b> (student, age {age})"
                f"<br>Sol A predicted: {sp_pred} / {pop_pred}")
    return (f"<b>{row['IID']}</b> (1000G reference)"
            f"<br>{row['superpopulation']} / {row['population']}"
            f"<br>{row['population_name']}")
proj["hover"] = proj.apply(hover, axis=1)

studs = proj[proj["is_student"]]

# 3D projected PCA
fig3d = go.Figure()
for sp in ["AFR", "AMR", "EAS", "EUR", "SAS"]:
    sub = proj[(~proj["is_student"]) & (proj["superpopulation"] == sp)]
    fig3d.add_trace(go.Scatter3d(
        x=sub["PC1"], y=sub["PC2"], z=sub["PC3"], mode="markers",
        marker=dict(size=4, color=SUPERPOP_COLORS[sp], opacity=0.75),
        name=sp, hovertext=sub["hover"], hoverinfo="text",
    ))
fig3d.add_trace(go.Scatter3d(
    x=studs["PC1"], y=studs["PC2"], z=studs["PC3"], mode="markers",
    marker=dict(size=6, color="white", symbol="diamond",
                line=dict(width=1.4, color="black")),
    name="students (projected)", hovertext=studs["hover"], hoverinfo="text",
))
fig3d.update_layout(
    title="GRM-projected 3D PCA — students projected onto reference axes (LASER-style)",
    scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
               bgcolor="#0e1116",
               xaxis=dict(gridcolor="#2a2f3a", color="#ccc", backgroundcolor="#0e1116"),
               yaxis=dict(gridcolor="#2a2f3a", color="#ccc", backgroundcolor="#0e1116"),
               zaxis=dict(gridcolor="#2a2f3a", color="#ccc", backgroundcolor="#0e1116")),
    paper_bgcolor="#0e1116", plot_bgcolor="#0e1116",
    font=dict(color="#eee"), height=680, margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=0, x=0.5, xanchor="center",
                bgcolor="rgba(0,0,0,0)"),
)

# 2D PC1/PC2 scatter (interactive)
fig2d = go.Figure()
for sp in ["AFR", "AMR", "EAS", "EUR", "SAS"]:
    sub = proj[(~proj["is_student"]) & (proj["superpopulation"] == sp)]
    fig2d.add_trace(go.Scatter(
        x=sub["PC1"], y=sub["PC2"], mode="markers",
        marker=dict(size=10, color=SUPERPOP_COLORS[sp], opacity=0.75),
        name=sp, hovertext=sub["hover"], hoverinfo="text",
    ))
fig2d.add_trace(go.Scatter(
    x=studs["PC1"], y=studs["PC2"], mode="markers+text",
    marker=dict(size=14, color="white", symbol="diamond",
                line=dict(width=1.4, color="black")),
    text=studs["IID"], textposition="top center", textfont=dict(size=8, color="#eee"),
    name="students (projected)", hovertext=studs["hover"], hoverinfo="text",
))
fig2d.update_layout(
    title="GRM-projected PCA — PC1 / PC2",
    paper_bgcolor="#0e1116", plot_bgcolor="#0e1116",
    font=dict(color="#eee"), height=620, margin=dict(l=50, r=20, t=60, b=40),
    xaxis=dict(title="PC1", gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"),
    yaxis=dict(title="PC2", gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"),
)

def fdiv(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       config={"displaylogo": False, "responsive": True})

# Stats
F_df = pd.read_csv(RES / "grm_F_inbreeding.tsv", sep="\t")
grm_pairs = pd.read_csv(RES / "grm_related_pairs.tsv", sep="\t")
cmp_pairs = pd.read_csv(RES / "grm_vs_king_pairs.tsv", sep="\t")
king_pairs = pd.read_csv(RES / "kinship_pairs.tsv", sep="\t")
r = float(np.corrcoef(cmp_pairs["GRM"], cmp_pairs["KING"])[0, 1])

stud_only = grm_pairs[
    (~grm_pairs["ID1"].str.startswith("ref")) &
    (~grm_pairs["ID2"].str.startswith("ref"))]
king_set = set(frozenset([r0["ID1"], r0["ID2"]]) for _, r0 in king_pairs.iterrows())
grm_set = set(frozenset([r0["ID1"], r0["ID2"]]) for _, r0 in stud_only.iterrows())

# Standalone Solution B page
html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Solution B — GRM-based pipeline · Digital Genealogist</title>
<script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
<style>
  :root {{
    --bg:#0a0d12; --panel:#0f141c; --panel-2:#151b24;
    --border:#1f2733; --border-2:#232b39;
    --text:#e7e9ee; --text-dim:#9aa3b2; --text-dim-2:#566273;
    --accent:#7cd0ff;
  }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text);
         font-family:-apple-system,"Segoe UI",Roboto,sans-serif;
         margin:0; padding:0; }}
  header {{ padding:36px 40px 0; max-width:1200px; margin:0 auto; }}
  header .crumb {{ color:var(--text-dim-2); font-size:12px; margin-bottom:8px; }}
  header .crumb a {{ color:var(--text-dim); text-decoration:none; }}
  header .crumb a:hover {{ color:var(--accent); }}
  h1 {{ font-size:30px; font-weight:300; margin:0 0 6px 0; }}
  h1 b {{ color:var(--accent); font-weight:600; }}
  header .sub {{ color:var(--text-dim); margin-bottom:24px; max-width:800px; }}
  main {{ max-width:1200px; margin:0 auto; padding:0 40px 40px; }}
  section {{ background:var(--panel); border:1px solid var(--border);
            border-radius:14px; padding:22px; margin-bottom:24px; }}
  section h3 {{ margin:0 0 8px 0; color:var(--accent); font-weight:500; font-size:18px; }}
  section .note {{ color:var(--text-dim); font-size:13px; margin:0 0 16px 0; }}
  img.figure {{ width:100%; border-radius:8px; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
           gap:14px; margin-bottom:24px; }}
  .stat {{ background:var(--panel-2); border:1px solid var(--border-2);
          padding:14px 18px; border-radius:10px; }}
  .stat .v {{ font-size:24px; font-weight:600; color:#fff; line-height:1; }}
  .stat .l {{ font-size:10px; color:var(--text-dim); margin-top:8px;
             text-transform:uppercase; letter-spacing:1px; }}
  table.cmp {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:13px; }}
  table.cmp th, table.cmp td {{ padding:9px 12px; border-bottom:1px solid var(--border);
                                text-align:left; }}
  table.cmp th {{ color:var(--accent); font-weight:500; }}
  table.cmp tr:hover td {{ background:rgba(124,208,255,0.04); }}
  table.cmp td:nth-child(2) {{ color:#9be57b; }}
  table.cmp td:nth-child(3) {{ color:#ffb27a; }}
  code {{ background:var(--panel-2); padding:2px 6px; border-radius:4px;
         font-size:0.9em; }}
  footer {{ color:var(--text-dim-2); font-size:11px; padding:24px;
           text-align:center; }}
  .back {{ display:inline-block; color:var(--accent); text-decoration:none;
          margin-bottom:18px; font-size:13px; }}
  .back:hover {{ text-decoration:underline; }}
</style></head>
<body>

<header>
  <div class="crumb"><a href="index.html">← Digital Genealogist</a> · Solution B</div>
  <h1>Solution <b>B</b> — GRM-based pipeline</h1>
  <p class="sub">
    Alternative reconstruction using the <b>Genetic Relatedness Matrix</b>
    (<code>plink2 --make-rel</code>, VanRaden 2008). Same input data, different
    approach: PCs from GRM eigendecomposition, students projected onto
    reference axes (LASER-style), kinship and inbreeding directly from GRM
    entries. Cross-checked against Solution A (KING + plink --pca + RF/NNLS).
  </p>
</header>

<main>

<section>
  <h3>How it differs from Solution A</h3>
  <table class="cmp">
    <thead><tr><th>Step</th><th>Solution A</th><th>Solution B (this page)</th></tr></thead>
    <tbody>
      <tr><td>Kinship metric</td>
          <td>KING-robust (population-stratification-resistant)</td>
          <td>VanRaden GRM (inflated by population structure)</td></tr>
      <tr><td>PCA</td>
          <td>Joint <code>plink2 --pca</code> on all 278 samples</td>
          <td>Eigendecomposition of <i>ref-only</i> GRM, students projected</td></tr>
      <tr><td>Inbreeding F</td>
          <td>not computed</td>
          <td>F<sub>i</sub> = GRM<sub>ii</sub> − 1</td></tr>
      <tr><td>Related pair detection</td>
          <td>127 KING pairs (cross-population safe)</td>
          <td>{len(grm_set):,} student-student GRM pairs ({len(king_set):,} match KING)</td></tr>
      <tr><td>Student influence on PC axes</td>
          <td>yes (joint PCA)</td>
          <td><b>no</b> (reference projection)</td></tr>
    </tbody>
  </table>
</section>

<div class="stats">
  <div class="stat"><div class="v">208</div><div class="l">refs in GRM</div></div>
  <div class="stat"><div class="v">70</div><div class="l">students projected</div></div>
  <div class="stat"><div class="v">{len(grm_set):,}</div><div class="l">GRM student-pairs G&gt;0.044</div></div>
  <div class="stat"><div class="v">{len(king_set & grm_set):,}</div><div class="l">KING ∩ GRM</div></div>
  <div class="stat"><div class="v">{r:.3f}</div><div class="l">Pearson r (KING vs GRM)</div></div>
  <div class="stat"><div class="v">{F_df['F_inbreeding'].max():.2f}</div><div class="l">max F<sub>inbreeding</sub></div></div>
</div>

<section>
  <h3>3D projected PCA (interactive)</h3>
  <p class="note">Top 3 eigenvectors of the <b>reference-only</b> 208×208 GRM.
  Students projected: PC<sub>k</sub>(s) = (1/√λ<sub>k</sub>) Σ<sub>i∈ref</sub>
  v<sub>ki</sub> · GRM(s,i). Drag to rotate.</p>
  {fdiv(fig3d)}
</section>

<section>
  <h3>PC1 / PC2 with student labels</h3>
  {fdiv(fig2d)}
</section>

<section>
  <h3>Joint GRM heatmap (278 × 278)</h3>
  <p class="note">Refs + students, ordered by superpopulation. Bright spots inside the
  bottom-right STUDENT block reveal family relations (parent–child, MZ-twins).
  Off-block brightness shows each student's closest reference superpopulation.</p>
  <img class="figure" src="assets/grm_joint_heatmap.png" alt="Joint GRM heatmap">
</section>

<section>
  <h3>Solution A vs Solution B principal components</h3>
  <p class="note">PC1 and PC2 are essentially identical (r&nbsp;≈&nbsp;0.96–0.99). PC3 differs
  (r&nbsp;=&nbsp;0.05) — joint PCA's PC3 picked up student-cohort variance, while
  Solution B's PC3 is purely a 1000G axis. This is exactly why
  reference-projection is preferred when the new cohort might be admixed.</p>
  <img class="figure" src="assets/grm_vs_plink_pca.png" alt="PCA comparison">
</section>

<section>
  <h3>GRM vs KING — pairwise kinship</h3>
  <p class="note">2,415 student-student pairs. The left vertical strip
  (KING&nbsp;=&nbsp;0, GRM up to +0.27) is population-structure inflation in GRM:
  unrelated pairs from the same superpopulation get a positive GRM value.
  The diagonal trend (KING&nbsp;~&nbsp;0.10–0.25, GRM&nbsp;~&nbsp;0.3–0.7) shows
  GRM&nbsp;≈&nbsp;2&nbsp;×&nbsp;kinship for actual relatives. The three top-right
  outliers at KING&nbsp;=&nbsp;0.5 are the MZ-twin pairs.</p>
  <img class="figure" src="assets/grm_vs_king_scatter.png" alt="GRM vs KING">
</section>

<section>
  <h3>F<sub>inbreeding</sub> per sample</h3>
  <p class="note">F<sub>i</sub> = GRM<sub>ii</sub> − 1. Warning: this estimator is
  biased by population structure — AFR samples appear "more inbred" simply
  because their allele frequencies deviate from the joint mean. A bias-free
  F requires either per-population allele frequencies or ROH-based F.</p>
  <img class="figure" src="assets/grm_F_inbreeding.png" alt="F coefficients">
</section>

<section>
  <h3>Takeaway</h3>
  <p class="note">
    <b>Use KING for kinship</b>, <b>use GRM-projection for PCA when the new cohort
    is small or admixed</b>. The two pipelines agree on every actual relative
    (KING&nbsp;⊆&nbsp;GRM); GRM additionally over-calls because of population
    structure, which is exactly the bias KING-robust was designed to remove
    (Manichaikul&nbsp;et&nbsp;al.&nbsp;2010).
  </p>
</section>

</main>
<footer>
  Pipeline: <code>plink2 --make-rel</code> → NumPy eigendecomposition + projection.
  Data: 181k LD-pruned SNPs · 208 1000G refs · 70 students.
</footer>

</body></html>"""

(SITE / "solution_b.html").write_text(html, encoding="utf-8")
print(f"Wrote site/solution_b.html ({(SITE/'solution_b.html').stat().st_size/1024:.0f} KB)")

# Add link from main index.html to solution_b.html
idx_path = SITE / "index.html"
idx = idx_path.read_text(encoding="utf-8")
banner = """<div style="background:linear-gradient(90deg,#3a2d4a,#1f2733);border:1px solid #4a3d5a;
  padding:14px 22px;border-radius:10px;margin-bottom:20px;display:flex;align-items:center;
  justify-content:space-between;flex-wrap:wrap;gap:12px;">
  <div style="color:#e7e9ee;font-size:14px;">
    <b style="color:#c98ff0;">Solution B</b> — alternative GRM-based pipeline with reference-projected PCA
    available
  </div>
  <a href="solution_b.html" style="background:#7cd0ff;color:#0a0d12;padding:8px 16px;
     border-radius:6px;text-decoration:none;font-weight:600;font-size:13px;">
    View Solution B →
  </a>
</div>"""
if "solution_b.html" not in idx:
    # Insert just after <main>
    idx = idx.replace("<main>\n\n", "<main>\n\n" + banner + "\n\n", 1)
    idx_path.write_text(idx, encoding="utf-8")
    print("Inserted banner into index.html")
else:
    print("Banner already present.")
