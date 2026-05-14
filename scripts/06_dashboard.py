"""Build a self-contained interactive HTML dashboard with all key views.

Sections (each a Plotly panel):
  1. 3D PCA (PC1/2/3) rotatable, color=superpop, student rendered as black diamond,
     hover shows id/age/pop/ancestry-bar.
  2. 2D PCA PC1/2 with custom hover (incl. ancestry bar in HTML).
  3. UMAP of PC1-10 with hover.
  4. Stacked-bar global ancestry with sortable students.
  5. Sunburst: superpop -> population -> student.
  6. Kinship network (plotly scatter with lines).

All assembled in a single index.html using Plotly's HTML embedding.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import networkx as nx

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# --- Load everything ---
pca = pd.read_csv(RES / "pca_coords.tsv", sep="\t")
summary = pd.read_csv(RES / "student_summary.tsv", sep="\t")
kin = pd.read_csv(RES / "kinship_pairs.tsv", sep="\t")
anc = pd.read_csv(RES / "ancestry_proportions.tsv", sep="\t")

SUPERPOP_COLORS = {
    "AFR": "#e41a1c", "EUR": "#377eb8", "EAS": "#4daf4a",
    "SAS": "#984ea3", "AMR": "#ff7f00", "STUDENT": "#111111",
}

# Drop existing 'age'/'sample_id' to avoid suffix collision
pca = pca.drop(columns=[c for c in ["age", "sample_id", "sample_id_fam"] if c in pca.columns])

# Compute UMAP if missing
if "UMAP1" not in pca.columns:
    import umap as _umap
    N_PC = 10
    X = pca[[f"PC{i+1}" for i in range(N_PC)]].values
    emb = _umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42).fit_transform(X)
    pca["UMAP1"] = emb[:, 0]
    pca["UMAP2"] = emb[:, 1]
# Merge ancestry + summary into pca for hover info
pca = pca.merge(summary[["sample_id", "age", "family_id", "pred_pop1", "pred_pop1_name",
                         "anc_AFR", "anc_AMR", "anc_EAS", "anc_EUR", "anc_SAS"]],
                left_on="IID", right_on="sample_id", how="left")

def ancestry_hover_html(row):
    parts = []
    for sp in ["AFR", "AMR", "EAS", "EUR", "SAS"]:
        v = row.get(f"anc_{sp}", np.nan)
        if pd.notna(v) and v > 0.001:
            bar = "█" * max(1, int(round(v * 20)))
            parts.append(f"<span style='color:{SUPERPOP_COLORS[sp]}'>{sp}: {bar} {v:.2f}</span>")
    return "<br>".join(parts) if parts else ""

def hover_text(row):
    is_stud = row["is_student"]
    base = f"<b>{row['IID']}</b>"
    if is_stud:
        base += f" (student, age {int(row['age'])})"
        base += f"<br>Family: #{int(row['family_id'])}"
        base += f"<br>Predicted: <b>{row['superpopulation']}</b> / {row['pred_pop1']} ({row['pred_pop1_name']})"
        anc_html = ancestry_hover_html(row)
        if anc_html:
            base += "<br>Ancestry:<br>" + anc_html
    else:
        base += f" (reference)<br>{row['superpopulation']} / {row['population']}"
        if isinstance(row.get("population_name"), str):
            base += f"<br>{row['population_name']}"
    return base

# Hover column - need population_name for refs
ref_meta = pd.read_csv(ROOT / "student_reference_metadata.tsv", sep="\t")
pop_to_name = ref_meta.set_index("population")["population_name"].to_dict()
pca["population_name"] = pca["population"].map(pop_to_name)
pca["hover"] = pca.apply(hover_text, axis=1)

# ---- 1. 3D PCA ----
fig3d = go.Figure()
for sp in ["AFR", "AMR", "EAS", "EUR", "SAS"]:
    sub = pca[(~pca["is_student"]) & (pca["superpopulation"] == sp)]
    fig3d.add_trace(go.Scatter3d(
        x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
        mode="markers",
        marker=dict(size=4, color=SUPERPOP_COLORS[sp], opacity=0.75,
                    line=dict(width=0)),
        name=sp,
        hovertext=sub["hover"], hoverinfo="text",
        legendgroup=sp,
    ))
studs = pca[pca["is_student"]]
fig3d.add_trace(go.Scatter3d(
    x=studs["PC1"], y=studs["PC2"], z=studs["PC3"],
    mode="markers",
    marker=dict(size=6, color="black", symbol="diamond",
                line=dict(width=1.2, color="white")),
    name="students",
    hovertext=studs["hover"], hoverinfo="text",
))
fig3d.update_layout(
    title="<b>3D PCA</b> — drag to rotate, hover for details",
    scene=dict(xaxis_title="PC1 (27.7%)", yaxis_title="PC2 (16.4%)", zaxis_title="PC3 (7.1%)",
               bgcolor="#0e1116",
               xaxis=dict(gridcolor="#2a2f3a", color="#ccc", backgroundcolor="#0e1116"),
               yaxis=dict(gridcolor="#2a2f3a", color="#ccc", backgroundcolor="#0e1116"),
               zaxis=dict(gridcolor="#2a2f3a", color="#ccc", backgroundcolor="#0e1116")),
    paper_bgcolor="#0e1116", plot_bgcolor="#0e1116",
    font=dict(color="#eee"), height=620, margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=0, x=0.5, xanchor="center",
                bgcolor="rgba(0,0,0,0)"),
)

# ---- 2. 2D PCA (4 subplots: PC1/2 colored by superpop, pop, family, ancestry-dominant) ----
from plotly.subplots import make_subplots

fig_pca2d = make_subplots(rows=1, cols=2, subplot_titles=("PC1 vs PC2 — by superpopulation",
                                                          "PC3 vs PC4 — by superpopulation"))
for col_i, (xc, yc) in enumerate([("PC1","PC2"), ("PC3","PC4")], start=1):
    for sp in ["AFR", "AMR", "EAS", "EUR", "SAS"]:
        sub = pca[(~pca["is_student"]) & (pca["superpopulation"] == sp)]
        fig_pca2d.add_trace(go.Scatter(
            x=sub[xc], y=sub[yc], mode="markers",
            marker=dict(size=8, color=SUPERPOP_COLORS[sp], opacity=0.7, line=dict(width=0)),
            name=sp, hovertext=sub["hover"], hoverinfo="text",
            legendgroup=sp, showlegend=(col_i==1),
        ), row=1, col=col_i)
    fig_pca2d.add_trace(go.Scatter(
        x=studs[xc], y=studs[yc], mode="markers",
        marker=dict(size=11, color="black", symbol="diamond",
                    line=dict(width=1.2, color="white")),
        name="students", hovertext=studs["hover"], hoverinfo="text",
        legendgroup="students", showlegend=(col_i==1),
    ), row=1, col=col_i)
fig_pca2d.update_layout(
    title="<b>2D PCA</b>",
    paper_bgcolor="#0e1116", plot_bgcolor="#0e1116", font=dict(color="#eee"),
    height=520, margin=dict(l=40, r=10, t=60, b=40),
)
fig_pca2d.update_xaxes(gridcolor="#2a2f3a", zerolinecolor="#2a2f3a")
fig_pca2d.update_yaxes(gridcolor="#2a2f3a", zerolinecolor="#2a2f3a")

# ---- 3. UMAP ----
fig_umap = go.Figure()
for sp in ["AFR", "AMR", "EAS", "EUR", "SAS"]:
    sub = pca[(~pca["is_student"]) & (pca["superpopulation"] == sp)]
    fig_umap.add_trace(go.Scatter(
        x=sub["UMAP1"], y=sub["UMAP2"], mode="markers",
        marker=dict(size=9, color=SUPERPOP_COLORS[sp], opacity=0.7),
        name=sp, hovertext=sub["hover"], hoverinfo="text",
    ))
fig_umap.add_trace(go.Scatter(
    x=studs["UMAP1"], y=studs["UMAP2"], mode="markers+text",
    marker=dict(size=12, color="black", symbol="diamond",
                line=dict(width=1.2, color="white")),
    text=studs["IID"], textposition="top center",
    textfont=dict(size=8, color="#eee"),
    name="students", hovertext=studs["hover"], hoverinfo="text",
))
fig_umap.update_layout(
    title="<b>UMAP</b> of first 10 PCs",
    paper_bgcolor="#0e1116", plot_bgcolor="#0e1116", font=dict(color="#eee"),
    height=620, margin=dict(l=40, r=10, t=60, b=40),
    xaxis=dict(title="UMAP1", gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"),
    yaxis=dict(title="UMAP2", gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"),
)

# ---- 4. Stacked ancestry bar (sorted by dominant) ----
anc_df = summary[["sample_id", "age", "family_id", "pred_superpop",
                  "anc_AFR", "anc_AMR", "anc_EAS", "anc_EUR", "anc_SAS"]].copy()
sp_cols = ["AFR", "AMR", "EAS", "EUR", "SAS"]
anc_df["dom"] = anc_df[[f"anc_{s}" for s in sp_cols]].idxmax(axis=1).str.replace("anc_", "")
anc_df["dom_val"] = anc_df[[f"anc_{s}" for s in sp_cols]].max(axis=1)
order = {"EUR": 0, "EAS": 1, "SAS": 2, "AMR": 3, "AFR": 4}
anc_df["dom_ord"] = anc_df["dom"].map(order)
anc_df = anc_df.sort_values(["dom_ord", "dom_val"], ascending=[True, False]).reset_index(drop=True)

fig_anc = go.Figure()
for sp in sp_cols:
    fig_anc.add_trace(go.Bar(
        x=anc_df["sample_id"], y=anc_df[f"anc_{sp}"],
        name=sp, marker_color=SUPERPOP_COLORS[sp], marker_line_width=0,
        hovertemplate=f"<b>%{{x}}</b><br>{sp}: %{{y:.3f}}<extra></extra>",
    ))
fig_anc.update_layout(
    title="<b>Global ancestry</b> — supervised NNLS (181k SNPs)",
    barmode="stack",
    paper_bgcolor="#0e1116", plot_bgcolor="#0e1116", font=dict(color="#eee"),
    xaxis=dict(title="", tickangle=-90, tickfont=dict(size=9)),
    yaxis=dict(title="Ancestry proportion", range=[0, 1], gridcolor="#2a2f3a"),
    height=520, margin=dict(l=50, r=10, t=60, b=80), bargap=0.05,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
)

# ---- 5. Sunburst: superpop -> population -> student ----
sun_rows = []
for _, r in summary.iterrows():
    sun_rows.append({
        "ids": f"{r['pred_superpop']}/{r['pred_pop1']}/{r['sample_id']}",
        "parents": f"{r['pred_superpop']}/{r['pred_pop1']}",
        "labels": r["sample_id"], "value": 1,
        "color": SUPERPOP_COLORS[r["pred_superpop"]],
    })
# population rollups (value must equal sum of children for branchvalues=total)
for (sp, pop), grp in summary.groupby(["pred_superpop", "pred_pop1"]):
    sun_rows.append({
        "ids": f"{sp}/{pop}", "parents": sp, "labels": pop, "value": len(grp),
        "color": SUPERPOP_COLORS[sp],
    })
for sp, grp in summary.groupby("pred_superpop"):
    sun_rows.append({"ids": sp, "parents": "", "labels": sp, "value": len(grp),
                     "color": SUPERPOP_COLORS[sp]})
sun = pd.DataFrame(sun_rows)
fig_sun = go.Figure(go.Sunburst(
    ids=sun["ids"], parents=sun["parents"], labels=sun["labels"], values=sun["value"],
    marker=dict(colors=sun["color"]),
    branchvalues="total",
    insidetextorientation="auto",
    hovertemplate="<b>%{label}</b><br>%{value:.0f}<extra></extra>",
))
fig_sun.update_layout(
    title="<b>Sunburst</b> — students grouped by predicted superpop → population",
    paper_bgcolor="#0e1116", font=dict(color="#eee"),
    height=620, margin=dict(l=10, r=10, t=60, b=10),
)

# ---- 6. Kinship network ----
G = nx.Graph()
students_set = set(summary["sample_id"])
for sid in students_set:
    r = summary[summary["sample_id"] == sid].iloc[0]
    G.add_node(sid, age=r["age"], family=r["family_id"], superpop=r["pred_superpop"])
REL_COLORS = {"MZ/dup": "#f93838", "parent-child": "#4fc3f7",
              "full-sibling": "#7be57b", "2nd-degree": "#ffb74d",
              "3rd-degree": "#888"}
for _, r in kin.iterrows():
    G.add_edge(r["ID1"], r["ID2"], rel=r["relation"], kin=r["KINSHIP"])
pos = nx.spring_layout(G, seed=42, k=0.6, iterations=200)

# Edges by type
edge_traces = []
for rel in ["MZ/dup", "parent-child", "full-sibling", "2nd-degree", "3rd-degree"]:
    ex, ey = [], []
    for u, v, d in G.edges(data=True):
        if d["rel"] != rel:
            continue
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    if not ex:
        continue
    edge_traces.append(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(color=REL_COLORS[rel], width=3 if rel in {"MZ/dup", "parent-child"} else 1.8),
        name=rel, hoverinfo="none",
    ))
# Nodes
node_x = [pos[n][0] for n in G.nodes]
node_y = [pos[n][1] for n in G.nodes]
node_color = [SUPERPOP_COLORS.get(G.nodes[n].get("superpop", "?"), "#999") for n in G.nodes]
node_text = []
for n in G.nodes:
    a = G.nodes[n]["age"]; f = G.nodes[n]["family"]; sp = G.nodes[n]["superpop"]
    node_text.append(f"<b>{n}</b><br>age {int(a)} · family #{int(f)} · {sp}")
fig_net = go.Figure(edge_traces + [
    go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=18, color=node_color, line=dict(color="white", width=1.2)),
        text=[n for n in G.nodes], textposition="middle center",
        textfont=dict(color="white", size=8),
        hovertext=node_text, hoverinfo="text", name="students",
    )
])
fig_net.update_layout(
    title="<b>Kinship network</b> — node color = superpop, edge color = relation type",
    paper_bgcolor="#0e1116", plot_bgcolor="#0e1116", font=dict(color="#eee"),
    height=720, margin=dict(l=10, r=10, t=60, b=10),
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    legend=dict(orientation="h", yanchor="bottom", y=0, x=0.5, xanchor="center"),
)

# ---- Assemble HTML dashboard ----
pio.templates.default = "plotly_dark"

def fig_div(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       config={"displaylogo": False, "responsive": True})

# Stats
n_students = len(summary)
n_families = summary["family_id"].max()
n_mz = (kin["relation"] == "MZ/dup").sum()
n_pc = (kin["relation"] == "parent-child").sum()
n_sib = (kin["relation"] == "full-sibling").sum()
n_admixed = (summary[["anc_AFR","anc_AMR","anc_EAS","anc_EUR","anc_SAS"]].max(axis=1) < 0.75).sum()

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Digital Genealogist — interactive dashboard</title>
<script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
<style>
  body {{ background:#0a0d12; color:#e7e9ee; font-family:-apple-system,Segoe UI,Roboto,sans-serif;
         margin:0; padding:24px; }}
  h1 {{ font-weight:300; font-size:28px; margin:0 0 4px 0; letter-spacing:.5px; }}
  h1 b {{ color:#7cd0ff; font-weight:600; }}
  .sub {{ color:#9aa3b2; margin-bottom:24px; font-size:13px; }}
  .stats {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:28px; }}
  .stat {{ background:#151b24; border:1px solid #232b39; padding:14px 18px; border-radius:10px; min-width:130px;
          box-shadow:0 1px 0 #0e1218; }}
  .stat .v {{ font-size:26px; font-weight:600; color:#fff; line-height:1; }}
  .stat .l {{ font-size:11px; color:#8a93a3; margin-top:6px; letter-spacing:.5px; text-transform:uppercase; }}
  .stat .v.afr {{ color:#ff7676 }} .stat .v.eur {{ color:#7cb8ff }} .stat .v.eas {{ color:#7be57b }}
  .stat .v.sas {{ color:#c98ff0 }} .stat .v.amr {{ color:#ffb066 }}
  .panel {{ background:#0f141c; border:1px solid #1f2733; border-radius:14px; padding:18px;
           margin-bottom:24px; box-shadow:0 1px 0 #060a10; }}
  .row {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
  @media (max-width:1100px) {{ .row {{ grid-template-columns:1fr; }} }}
  .legend-rel {{ display:flex; gap:14px; font-size:12px; color:#9aa3b2; margin-top:6px; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }}
  footer {{ color:#566273; font-size:11px; margin-top:30px; text-align:center; }}
  a {{ color:#7cd0ff; text-decoration:none; }}
</style></head>
<body>
<h1>Digital <b>Genealogist</b> — 70 students, 208 references, 5M variants</h1>
<div class="sub">PCA · supervised classification · KING kinship · NNLS ancestry · 1000 Genomes Project reference panel</div>

<div class="stats">
  <div class="stat"><div class="v">{n_students}</div><div class="l">students</div></div>
  <div class="stat"><div class="v">{n_families}</div><div class="l">families</div></div>
  <div class="stat"><div class="v">{n_pc}</div><div class="l">parent–child links</div></div>
  <div class="stat"><div class="v">{n_sib}</div><div class="l">full-sibling links</div></div>
  <div class="stat"><div class="v">{n_mz}</div><div class="l">MZ-twin pairs</div></div>
  <div class="stat"><div class="v">{n_admixed}</div><div class="l">admixed (max anc&lt;.75)</div></div>
  <div class="stat"><div class="v eur">{(summary['pred_superpop']=='EUR').sum()}</div><div class="l">EUR</div></div>
  <div class="stat"><div class="v sas">{(summary['pred_superpop']=='SAS').sum()}</div><div class="l">SAS</div></div>
  <div class="stat"><div class="v afr">{(summary['pred_superpop']=='AFR').sum()}</div><div class="l">AFR</div></div>
  <div class="stat"><div class="v eas">{(summary['pred_superpop']=='EAS').sum()}</div><div class="l">EAS</div></div>
  <div class="stat"><div class="v amr">{(summary['pred_superpop']=='AMR').sum()}</div><div class="l">AMR</div></div>
</div>

<div class="panel">{fig_div(fig3d)}</div>

<div class="row">
  <div class="panel">{fig_div(fig_pca2d)}</div>
  <div class="panel">{fig_div(fig_umap)}</div>
</div>

<div class="panel">{fig_div(fig_anc)}</div>

<div class="row">
  <div class="panel">{fig_div(fig_sun)}</div>
  <div class="panel">
    {fig_div(fig_net)}
    <div class="legend-rel">
      <span><span class="dot" style="background:#f93838"></span>MZ/duplicate</span>
      <span><span class="dot" style="background:#4fc3f7"></span>parent-child</span>
      <span><span class="dot" style="background:#7be57b"></span>full-sibling</span>
      <span><span class="dot" style="background:#ffb74d"></span>2nd-degree</span>
      <span><span class="dot" style="background:#888"></span>3rd-degree</span>
    </div>
  </div>
</div>

<div class="panel">
  <h3 style="margin-top:0;color:#7cd0ff;font-weight:400;">Radial family chart</h3>
  <p style="color:#9aa3b2;font-size:12px;margin-top:0;">All 70 students arranged on a circle, grouped by family. Outer arc = family. Pie = 5-way ancestry. Bezier curves inside = kinship relations.</p>
  <img src="radial_families.png" style="width:100%;border-radius:8px;">
</div>

<div class="panel">
  <h3 style="margin-top:0;color:#7cd0ff;font-weight:400;">Family pedigrees with ancestry pies</h3>
  <p style="color:#9aa3b2;font-size:12px;margin-top:0;">Y-axis = age (older at top). Each node is a pie of 5-way ancestry. Lines: solid blue = parent–child, solid green = full-sibling, thick red = MZ twin, dashed amber = 2nd degree.</p>
  <img src="family_pedigrees.png" style="width:100%;border-radius:8px;">
</div>

<div class="panel">
  <h3 style="margin-top:0;color:#7cd0ff;font-weight:400;">Hierarchically clustered ancestry</h3>
  <img src="ancestry_clustered.png" style="width:100%;border-radius:8px;">
</div>

<footer>
  Pipeline: plink2 (QC, LD-prune, PCA, KING) · Python (sklearn, scipy NNLS, networkx, umap, plotly).
  All artefacts in <code>results/</code>.
</footer>
</body></html>"""

(RES / "dashboard.html").write_text(html, encoding="utf-8")
print(f"Wrote {RES/'dashboard.html'}  ({(RES/'dashboard.html').stat().st_size/1024:.0f} KB)")
