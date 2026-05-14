"""Build a netlify-ready static site with ALL graphics in a single index.html.

Output:
  site/
    index.html      — single-page dashboard with sidebar nav, all interactive +
                      static visuals embedded
    assets/         — PNG images referenced from HTML
    data/           — TSV tables (downloadable)
    netlify.toml    — Netlify build config (no build, just publish ./)
    _headers        — Cache headers for assets
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import networkx as nx

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
RES = ROOT / "results"
SITE = ROOT / "site"
ASSETS = SITE / "assets"
DATA = SITE / "data"

# Fresh build — tolerate Windows file locks (Explorer/VSCode previews)
if SITE.exists():
    try:
        shutil.rmtree(SITE)
    except PermissionError:
        # Just delete files we'll overwrite; leave dir
        for p in SITE.rglob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except PermissionError:
                    pass
ASSETS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# Copy static PNGs
for png in ["radial_families.png", "family_pedigrees.png", "ancestry_clustered.png",
            "ancestry_barplot.png", "pca_superpop.png", "pca_population.png",
            "umap.png", "scree.png", "family_trees.png", "kinship_network.png"]:
    src = RES / png
    if src.exists():
        shutil.copy(src, ASSETS / png)
# family hi-res
(ASSETS / "families").mkdir()
for fp in (RES / "families").glob("*.png"):
    shutil.copy(fp, ASSETS / "families" / fp.name)

# Copy data tables
for tsv in ["student_summary.tsv", "population_predictions.tsv",
            "superpop_predictions.tsv", "ancestry_proportions.tsv",
            "kinship_pairs.tsv", "pca_coords.tsv"]:
    src = RES / tsv
    if src.exists():
        shutil.copy(src, DATA / tsv)
shutil.copy(RES / "REPORT.md", SITE / "REPORT.md")

# --- Build interactive figures (re-use logic from 06_dashboard.py) ---
SUPERPOP_COLORS = {
    "AFR": "#e41a1c", "EUR": "#377eb8", "EAS": "#4daf4a",
    "SAS": "#984ea3", "AMR": "#ff7f00", "STUDENT": "#111111",
}

pca = pd.read_csv(RES / "pca_coords.tsv", sep="\t")
summary = pd.read_csv(RES / "student_summary.tsv", sep="\t")
kin = pd.read_csv(RES / "kinship_pairs.tsv", sep="\t")
ref_meta = pd.read_csv(ROOT / "student_reference_metadata.tsv", sep="\t")
pop_to_name = ref_meta.set_index("population")["population_name"].to_dict()

# Drop existing 'age'/'sample_id' to avoid collision
pca = pca.drop(columns=[c for c in ["age", "sample_id", "sample_id_fam"] if c in pca.columns])
if "UMAP1" not in pca.columns:
    import umap as _umap
    X = pca[[f"PC{i+1}" for i in range(10)]].values
    emb = _umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42).fit_transform(X)
    pca["UMAP1"] = emb[:, 0]
    pca["UMAP2"] = emb[:, 1]
pca = pca.merge(summary[["sample_id", "age", "family_id", "pred_pop1", "pred_pop1_name",
                         "anc_AFR", "anc_AMR", "anc_EAS", "anc_EUR", "anc_SAS"]],
                left_on="IID", right_on="sample_id", how="left")
pca["population_name"] = pca["population"].map(pop_to_name)

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
pca["hover"] = pca.apply(hover_text, axis=1)
studs = pca[pca["is_student"]]

# === Figures ===
common_layout = dict(
    paper_bgcolor="#0e1116", plot_bgcolor="#0e1116",
    font=dict(color="#eee", family="-apple-system, Segoe UI, Roboto, sans-serif"),
    margin=dict(l=40, r=10, t=50, b=40),
)

# 1. 3D PCA
fig3d = go.Figure()
for sp in ["AFR", "AMR", "EAS", "EUR", "SAS"]:
    sub = pca[(~pca["is_student"]) & (pca["superpopulation"] == sp)]
    fig3d.add_trace(go.Scatter3d(
        x=sub["PC1"], y=sub["PC2"], z=sub["PC3"], mode="markers",
        marker=dict(size=4, color=SUPERPOP_COLORS[sp], opacity=0.75),
        name=sp, hovertext=sub["hover"], hoverinfo="text",
    ))
fig3d.add_trace(go.Scatter3d(
    x=studs["PC1"], y=studs["PC2"], z=studs["PC3"], mode="markers",
    marker=dict(size=6, color="black", symbol="diamond",
                line=dict(width=1.2, color="white")),
    name="students", hovertext=studs["hover"], hoverinfo="text",
))
fig3d.update_layout(
    title="3D PCA — drag to rotate, hover for details",
    scene=dict(xaxis_title="PC1 (27.7%)", yaxis_title="PC2 (16.4%)", zaxis_title="PC3 (7.1%)",
               bgcolor="#0e1116",
               xaxis=dict(gridcolor="#2a2f3a", color="#ccc", backgroundcolor="#0e1116"),
               yaxis=dict(gridcolor="#2a2f3a", color="#ccc", backgroundcolor="#0e1116"),
               zaxis=dict(gridcolor="#2a2f3a", color="#ccc", backgroundcolor="#0e1116")),
    **{**common_layout, "height": 680, "margin": dict(l=0, r=0, t=40, b=0)},
    legend=dict(orientation="h", yanchor="bottom", y=0, x=0.5, xanchor="center",
                bgcolor="rgba(0,0,0,0)"),
)

# 2. 2D PCA
fig_pca2d = make_subplots(rows=1, cols=2,
                          subplot_titles=("PC1 vs PC2", "PC3 vs PC4"))
for col_i, (xc, yc) in enumerate([("PC1","PC2"), ("PC3","PC4")], start=1):
    for sp in ["AFR", "AMR", "EAS", "EUR", "SAS"]:
        sub = pca[(~pca["is_student"]) & (pca["superpopulation"] == sp)]
        fig_pca2d.add_trace(go.Scatter(
            x=sub[xc], y=sub[yc], mode="markers",
            marker=dict(size=8, color=SUPERPOP_COLORS[sp], opacity=0.7),
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
fig_pca2d.update_xaxes(gridcolor="#2a2f3a", zerolinecolor="#2a2f3a")
fig_pca2d.update_yaxes(gridcolor="#2a2f3a", zerolinecolor="#2a2f3a")
fig_pca2d.update_layout(title="2D PCA — by superpopulation",
                        **{**common_layout, "height": 520, "margin": dict(l=40, r=10, t=60, b=40)})

# 3. UMAP
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
fig_umap.update_layout(title="UMAP of first 10 PCs",
                       **{**common_layout, "height": 620, "margin": dict(l=40, r=10, t=60, b=40)},
                       xaxis=dict(title="UMAP1", gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"),
                       yaxis=dict(title="UMAP2", gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"))

# 4. Stacked ancestry
sp_cols = ["AFR", "AMR", "EAS", "EUR", "SAS"]
anc_df = summary[["sample_id","age","family_id","pred_superpop"] + [f"anc_{s}" for s in sp_cols]].copy()
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
fig_anc.update_layout(title="Global ancestry — supervised NNLS (181k SNPs)",
                     barmode="stack",
                     **{**common_layout, "height": 520, "margin": dict(l=50, r=10, t=60, b=80)},
                     xaxis=dict(title="", tickangle=-90, tickfont=dict(size=9)),
                     yaxis=dict(title="Ancestry proportion", range=[0, 1], gridcolor="#2a2f3a"),
                     bargap=0.05,
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))

# 5. Sunburst — with correctly summed parent values for branchvalues="total"
sun_rows = []
# Leaves: students (value=1)
for _, r in summary.iterrows():
    sun_rows.append({
        "ids": f"{r['pred_superpop']}/{r['pred_pop1']}/{r['sample_id']}",
        "parents": f"{r['pred_superpop']}/{r['pred_pop1']}",
        "labels": r["sample_id"], "value": 1,
        "color": SUPERPOP_COLORS[r["pred_superpop"]],
    })
# Population rollups: value = number of students in that population
for (sp, pop), grp in summary.groupby(["pred_superpop", "pred_pop1"]):
    sun_rows.append({"ids": f"{sp}/{pop}", "parents": sp, "labels": pop, "value": len(grp),
                     "color": SUPERPOP_COLORS[sp]})
# Superpop rollups: value = number of students in that superpop
for sp, grp in summary.groupby("pred_superpop"):
    sun_rows.append({"ids": sp, "parents": "", "labels": sp, "value": len(grp),
                     "color": SUPERPOP_COLORS[sp]})
sun = pd.DataFrame(sun_rows)
fig_sun = go.Figure(go.Sunburst(
    ids=sun["ids"], parents=sun["parents"], labels=sun["labels"], values=sun["value"],
    marker=dict(colors=sun["color"]), branchvalues="total",
    insidetextorientation="auto",
    hovertemplate="<b>%{label}</b><br>%{value:.0f} student(s)<extra></extra>",
))
fig_sun.update_layout(title="Sunburst — students grouped by predicted superpop → population",
                     **{**common_layout, "height": 620, "margin": dict(l=10, r=10, t=60, b=10)})

# 6. Kinship network
G = nx.Graph()
for sid in summary["sample_id"]:
    r = summary[summary["sample_id"] == sid].iloc[0]
    G.add_node(sid, age=r["age"], family=r["family_id"], superpop=r["pred_superpop"])
REL_COLORS = {"MZ/dup": "#f93838", "parent-child": "#4fc3f7",
              "full-sibling": "#7be57b", "2nd-degree": "#ffb74d",
              "3rd-degree": "#888"}
for _, r in kin.iterrows():
    G.add_edge(r["ID1"], r["ID2"], rel=r["relation"], kin=r["KINSHIP"])
pos = nx.spring_layout(G, seed=42, k=0.6, iterations=200)
edge_traces = []
for rel in ["MZ/dup", "parent-child", "full-sibling", "2nd-degree", "3rd-degree"]:
    ex, ey = [], []
    for u, v, d in G.edges(data=True):
        if d["rel"] != rel: continue
        ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
    if not ex: continue
    edge_traces.append(go.Scatter(x=ex, y=ey, mode="lines",
        line=dict(color=REL_COLORS[rel], width=3 if rel in {"MZ/dup","parent-child"} else 1.8),
        name=rel, hoverinfo="none"))
node_x = [pos[n][0] for n in G.nodes]; node_y = [pos[n][1] for n in G.nodes]
node_color = [SUPERPOP_COLORS.get(G.nodes[n]["superpop"], "#999") for n in G.nodes]
node_text = [f"<b>{n}</b><br>age {int(G.nodes[n]['age'])} · family #{int(G.nodes[n]['family'])} · {G.nodes[n]['superpop']}" for n in G.nodes]
fig_net = go.Figure(edge_traces + [go.Scatter(
    x=node_x, y=node_y, mode="markers+text",
    marker=dict(size=18, color=node_color, line=dict(color="white", width=1.2)),
    text=list(G.nodes), textposition="middle center",
    textfont=dict(color="white", size=8),
    hovertext=node_text, hoverinfo="text", name="students",
)])
fig_net.update_layout(title="Kinship network",
                     **{**common_layout, "height": 720, "margin": dict(l=10, r=10, t=60, b=10)},
                     xaxis=dict(visible=False), yaxis=dict(visible=False),
                     legend=dict(orientation="h", yanchor="bottom", y=0, x=0.5, xanchor="center"))

# 7. Sankey
SP_ORDER = {"EUR": 0, "EAS": 1, "SAS": 2, "AMR": 3, "AFR": 4}
sp_set = sorted(summary["pred_superpop"].unique(), key=lambda s: SP_ORDER.get(s,99))
pop_set = list(summary.groupby(["pred_superpop","pred_pop1"]).groups.keys())
pop_labels = [p for (sp,p) in pop_set]
# family ordering
fam_order_data = []
for fid, grp in summary.groupby("family_id"):
    dom = grp[[f"anc_{s}" for s in sp_cols]].mean().idxmax().replace("anc_", "")
    fam_order_data.append((fid, dom, len(grp), SP_ORDER[dom]))
fam_order_data.sort(key=lambda t: (t[3], -t[2], t[0]))
fam_order = [t[0] for t in fam_order_data]
fam_color = {t[0]: SUPERPOP_COLORS[t[1]] for t in fam_order_data}
fam_labels = [f"#{f}" for f in fam_order]
labels = sp_set + pop_labels + fam_labels
node_colors = ([SUPERPOP_COLORS[s] for s in sp_set]
               + [SUPERPOP_COLORS[sp] for (sp,p) in pop_set]
               + [fam_color[f] for f in fam_order])
def idx_of(label, scope):
    if scope=="sp": return sp_set.index(label)
    if scope=="pop": return len(sp_set) + pop_set.index(label)
    if scope=="fam": return len(sp_set) + len(pop_set) + fam_order.index(label)
def hex_to_rgba(h,a):
    h=h.lstrip("#"); return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"
src, tgt, val, link_col = [], [], [], []
for (sp,pop), grp in summary.groupby(["pred_superpop","pred_pop1"]):
    src.append(idx_of(sp,"sp")); tgt.append(idx_of((sp,pop),"pop"))
    val.append(len(grp)); link_col.append(hex_to_rgba(SUPERPOP_COLORS[sp], 0.65))
for (sp,pop,fid), grp in summary.groupby(["pred_superpop","pred_pop1","family_id"]):
    src.append(idx_of((sp,pop),"pop")); tgt.append(idx_of(fid,"fam"))
    val.append(len(grp)); link_col.append(hex_to_rgba(SUPERPOP_COLORS[sp], 0.55))
fig_sankey = go.Figure(go.Sankey(
    node=dict(label=labels, color=node_colors, pad=15, thickness=18,
              line=dict(color="#0a0d12", width=0.5)),
    link=dict(source=src, target=tgt, value=val, color=link_col),
))
fig_sankey.update_layout(title="Superpopulation → Population → Family",
                        **{**common_layout, "height": 820, "margin": dict(l=10, r=10, t=60, b=20)})

# === Render to HTML ===
pio.templates.default = "plotly_dark"
def fdiv(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       config={"displaylogo": False, "responsive": True})

# Stats
n_students = len(summary)
n_families = summary["family_id"].max()
n_mz = (kin["relation"] == "MZ/dup").sum()
n_pc = (kin["relation"] == "parent-child").sum()
n_sib = (kin["relation"] == "full-sibling").sum()
n_2nd = (kin["relation"] == "2nd-degree").sum()
n_admixed = (summary[[f"anc_{s}" for s in sp_cols]].max(axis=1) < 0.75).sum()

# Build table HTML for summary
summary_disp = summary[[
    "sample_id","age","family_id","pred_superpop","pred_pop1","pred_pop1_name",
    "anc_AFR","anc_AMR","anc_EAS","anc_EUR","anc_SAS"
]].copy()
table_html = summary_disp.to_html(index=False, classes="data-table", border=0,
                                  float_format=lambda x: f"{x:.2f}")

html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Digital Genealogist · 70 students · 1000G reference panel</title>
<meta name="description" content="Interactive PCA, ancestry, and family pedigree dashboard from 5.9M variant VCF.">
<script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
<style>
  :root {{
    --bg: #0a0d12; --panel: #0f141c; --panel-2: #151b24;
    --border: #1f2733; --border-2: #232b39;
    --text: #e7e9ee; --text-dim: #9aa3b2; --text-dim-2: #566273;
    --accent: #7cd0ff;
    --afr: #ff7676; --eur: #7cb8ff; --eas: #7be57b; --sas: #c98ff0; --amr: #ffb066;
  }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text);
         font-family: -apple-system, "Segoe UI", Roboto, Helvetica, sans-serif;
         margin: 0; padding: 0; line-height: 1.5; }}
  nav.sidebar {{ position: fixed; top: 0; left: 0; width: 220px; height: 100vh;
                background: linear-gradient(180deg, #0d1218 0%, #0a0d12 100%);
                border-right: 1px solid var(--border); padding: 24px 18px;
                overflow-y: auto; z-index: 100; }}
  nav.sidebar h1 {{ font-size: 18px; font-weight: 300; margin: 0 0 4px 0; letter-spacing: .3px; }}
  nav.sidebar h1 b {{ color: var(--accent); font-weight: 600; }}
  nav.sidebar .sub {{ color: var(--text-dim-2); font-size: 11px; margin-bottom: 26px; }}
  nav.sidebar ul {{ list-style: none; padding: 0; margin: 0; }}
  nav.sidebar li {{ margin: 2px 0; }}
  nav.sidebar a {{ color: var(--text-dim); text-decoration: none; font-size: 13px;
                  display: block; padding: 7px 10px; border-radius: 6px;
                  transition: all .15s; border-left: 2px solid transparent; }}
  nav.sidebar a:hover {{ background: var(--panel-2); color: var(--text);
                        border-left-color: var(--accent); }}
  nav.sidebar .sec {{ color: var(--text-dim-2); font-size: 10px; text-transform: uppercase;
                     letter-spacing: 1px; margin: 18px 10px 6px; }}
  main {{ margin-left: 220px; padding: 28px 32px; max-width: 1500px; }}
  h2 {{ font-weight: 300; font-size: 26px; margin: 0 0 6px 0; color: #fff; letter-spacing: .3px; }}
  h2 .badge {{ background: var(--panel-2); color: var(--text-dim); padding: 3px 10px;
              border-radius: 12px; font-size: 12px; margin-left: 10px; vertical-align: middle; }}
  .lede {{ color: var(--text-dim); margin: 0 0 22px 0; font-size: 14px; max-width: 900px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
           gap: 14px; margin-bottom: 32px; }}
  .stat {{ background: var(--panel-2); border: 1px solid var(--border-2);
          padding: 14px 18px; border-radius: 10px; }}
  .stat .v {{ font-size: 26px; font-weight: 600; color: #fff; line-height: 1; }}
  .stat .l {{ font-size: 10px; color: var(--text-dim); margin-top: 8px;
             letter-spacing: 1px; text-transform: uppercase; }}
  .stat .v.afr {{ color: var(--afr); }} .stat .v.eur {{ color: var(--eur); }}
  .stat .v.eas {{ color: var(--eas); }} .stat .v.sas {{ color: var(--sas); }}
  .stat .v.amr {{ color: var(--amr); }}
  section {{ background: var(--panel); border: 1px solid var(--border);
            border-radius: 14px; padding: 22px; margin-bottom: 24px; scroll-margin-top: 16px; }}
  section h3 {{ margin: 0 0 4px 0; font-weight: 500; font-size: 18px;
               color: var(--accent); }}
  section .note {{ color: var(--text-dim); font-size: 12px; margin: 0 0 14px 0;
                  max-width: 900px; }}
  .row2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
  @media (max-width: 1200px) {{ .row2 {{ grid-template-columns: 1fr; }} }}
  @media (max-width: 800px) {{ nav.sidebar {{ position: static; width: 100%; height: auto; }}
                              main {{ margin-left: 0; padding: 16px; }} }}
  img.figure {{ width: 100%; border-radius: 8px; display: block; }}
  .legend-rel {{ display: flex; gap: 14px; font-size: 12px; color: var(--text-dim);
                margin-top: 8px; flex-wrap: wrap; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
         margin-right: 5px; vertical-align: middle; }}
  table.data-table {{ width: 100%; border-collapse: collapse; font-size: 12px;
                     margin-top: 10px; }}
  table.data-table th {{ background: var(--panel-2); color: var(--text);
                        text-align: left; padding: 8px 10px; border-bottom: 2px solid var(--border-2);
                        font-weight: 600; position: sticky; top: 0; z-index: 1; }}
  table.data-table td {{ padding: 6px 10px; border-bottom: 1px solid var(--border);
                        color: var(--text-dim); }}
  table.data-table tr:hover td {{ background: rgba(124,208,255,0.04); color: var(--text); }}
  .table-scroll {{ max-height: 600px; overflow-y: auto; border: 1px solid var(--border);
                  border-radius: 8px; }}
  .downloads a {{ color: var(--accent); text-decoration: none; margin-right: 18px;
                 font-size: 13px; }}
  .downloads a:hover {{ text-decoration: underline; }}
  footer {{ color: var(--text-dim-2); font-size: 11px; padding: 24px 0;
           text-align: center; border-top: 1px solid var(--border); margin-top: 30px; }}
  footer code {{ background: var(--panel-2); padding: 2px 6px; border-radius: 4px; }}
  .hero {{ background: radial-gradient(ellipse at top left, rgba(124,208,255,0.08), transparent 60%),
                       radial-gradient(ellipse at bottom right, rgba(255,118,118,0.05), transparent 60%),
                       var(--panel);
          padding: 32px; border-radius: 14px; margin-bottom: 28px;
          border: 1px solid var(--border); }}
  .hero h2 {{ font-size: 34px; margin-bottom: 8px; }}
  .hero p {{ font-size: 15px; color: var(--text-dim); max-width: 800px; }}
</style>
</head><body>

<nav class="sidebar">
  <h1>Digital <b>Genealogist</b></h1>
  <div class="sub">70 students · 1000G refs · 5.9M variants</div>
  <div class="sec">Overview</div>
  <ul>
    <li><a href="#overview">Overview & stats</a></li>
  </ul>
  <div class="sec">Population structure</div>
  <ul>
    <li><a href="#pca-3d">3D PCA (interactive)</a></li>
    <li><a href="#pca-2d">2D PCA</a></li>
    <li><a href="#umap">UMAP</a></li>
    <li><a href="#sunburst">Sunburst</a></li>
  </ul>
  <div class="sec">Ancestry</div>
  <ul>
    <li><a href="#ancestry-bar">Ancestry barplot</a></li>
    <li><a href="#ancestry-clust">Clustered ancestry</a></li>
    <li><a href="#sankey">Sankey flow</a></li>
  </ul>
  <div class="sec">Families & kinship</div>
  <ul>
    <li><a href="#radial">Radial family chart</a></li>
    <li><a href="#pedigrees">Pedigrees</a></li>
    <li><a href="#kinship">Kinship network</a></li>
  </ul>
  <div class="sec">Data</div>
  <ul>
    <li><a href="#table">Student summary table</a></li>
    <li><a href="#downloads">Downloads</a></li>
  </ul>
</nav>

<main>

<div id="overview" class="hero">
  <h2>Digital <b style="color:var(--accent);font-weight:600;">Genealogist</b></h2>
  <p>From a 14&nbsp;GB VCF of 278 samples and 5.9&nbsp;million variants, this pipeline reconstructs
  population structure, ancestry composition, and kinship relations among 70 anonymous students.
  Reference panel: 208 1000 Genomes samples across 5 superpopulations.</p>
  <div class="stats" style="margin-top: 24px; margin-bottom: 0;">
    <div class="stat"><div class="v">{n_students}</div><div class="l">students</div></div>
    <div class="stat"><div class="v">{n_families}</div><div class="l">families</div></div>
    <div class="stat"><div class="v">{n_pc}</div><div class="l">parent–child</div></div>
    <div class="stat"><div class="v">{n_sib}</div><div class="l">full-sibling</div></div>
    <div class="stat"><div class="v">{n_2nd}</div><div class="l">2nd-degree</div></div>
    <div class="stat"><div class="v">{n_mz}</div><div class="l">MZ-twin pairs</div></div>
    <div class="stat"><div class="v">{n_admixed}</div><div class="l">admixed (&lt;.75)</div></div>
    <div class="stat"><div class="v eur">{(summary['pred_superpop']=='EUR').sum()}</div><div class="l">EUR</div></div>
    <div class="stat"><div class="v sas">{(summary['pred_superpop']=='SAS').sum()}</div><div class="l">SAS</div></div>
    <div class="stat"><div class="v afr">{(summary['pred_superpop']=='AFR').sum()}</div><div class="l">AFR</div></div>
    <div class="stat"><div class="v eas">{(summary['pred_superpop']=='EAS').sum()}</div><div class="l">EAS</div></div>
    <div class="stat"><div class="v amr">{(summary['pred_superpop']=='AMR').sum()}</div><div class="l">AMR</div></div>
  </div>
</div>

<section id="pca-3d">
  <h3>3D PCA — interactive</h3>
  <p class="note">Principal components 1–3 computed on 181k LD-pruned SNPs (PLINK 2).
  References by superpopulation, students as black diamonds. Drag to rotate; hover for details.</p>
  {fdiv(fig3d)}
</section>

<section id="pca-2d">
  <h3>2D PCA — PC1/2 & PC3/4</h3>
  <p class="note">Top 27.7% of variance lies in PC1; major continental separation visible.</p>
  {fdiv(fig_pca2d)}
</section>

<section id="umap">
  <h3>UMAP of first 10 PCs</h3>
  <p class="note">Non-linear projection emphasising local cluster structure.
  Each student labelled by sample id.</p>
  {fdiv(fig_umap)}
</section>

<section id="sunburst">
  <h3>Sunburst — superpop → population → student</h3>
  <p class="note">Cascade classifier prediction. Click on a wedge to drill down.</p>
  {fdiv(fig_sun)}
</section>

<section id="ancestry-bar">
  <h3>Global ancestry (NNLS)</h3>
  <p class="note">Supervised non-negative least-squares fit of each student's genotype dosages
  against 1000G superpopulation allele frequencies (181k SNPs). Students sorted by dominant ancestry.</p>
  {fdiv(fig_anc)}
</section>

<section id="ancestry-clust">
  <h3>Hierarchically clustered ancestry</h3>
  <p class="note">Students ordered by Ward clustering on ancestry vectors. Tick label colour = family.</p>
  <img class="figure" src="assets/ancestry_clustered.png" alt="clustered ancestry">
</section>

<section id="sankey">
  <h3>Sankey: superpop → population → family</h3>
  <p class="note">Flow shows how predicted populations distribute into the recovered family clusters.</p>
  {fdiv(fig_sankey)}
</section>

<section id="radial">
  <h3>Radial family chart <span class="badge">signature view</span></h3>
  <p class="note">All 70 students on a circle, grouped by family. Outer arc colour = family
  dominant ancestry. Each node is a 5-way ancestry pie. Bezier curves inside the circle =
  kinship edges (red = MZ-twin, blue = parent-child, green = full-sibling, amber dashed = 2nd degree).</p>
  <img class="figure" src="assets/radial_families.png" alt="radial family chart">
</section>

<section id="pedigrees">
  <h3>Family pedigrees with ancestry pies</h3>
  <p class="note">Y-axis = age (older at top). Each node is a 5-way ancestry pie.
  Lines: solid blue = parent–child, solid green = full-sibling, thick red = MZ twin,
  dashed amber = 2nd degree.</p>
  <img class="figure" src="assets/family_pedigrees.png" alt="family pedigrees">
  <div class="legend-rel">
    <span><span class="dot" style="background:#ff3030"></span>MZ-twin</span>
    <span><span class="dot" style="background:#4fc3f7"></span>parent–child</span>
    <span><span class="dot" style="background:#7be57b"></span>full-sibling</span>
    <span><span class="dot" style="background:#ffb74d"></span>2nd-degree</span>
    <span><span class="dot" style="background:#e41a1c"></span>AFR</span>
    <span><span class="dot" style="background:#ff7f00"></span>AMR</span>
    <span><span class="dot" style="background:#4daf4a"></span>EAS</span>
    <span><span class="dot" style="background:#377eb8"></span>EUR</span>
    <span><span class="dot" style="background:#984ea3"></span>SAS</span>
  </div>
</section>

<section id="kinship">
  <h3>Kinship network</h3>
  <p class="note">Spring-layout graph; node colour = predicted superpopulation, edge colour = relation type.</p>
  {fdiv(fig_net)}
  <div class="legend-rel">
    <span><span class="dot" style="background:#f93838"></span>MZ/duplicate</span>
    <span><span class="dot" style="background:#4fc3f7"></span>parent–child</span>
    <span><span class="dot" style="background:#7be57b"></span>full-sibling</span>
    <span><span class="dot" style="background:#ffb74d"></span>2nd-degree</span>
    <span><span class="dot" style="background:#888"></span>3rd-degree</span>
  </div>
</section>

<section id="table">
  <h3>Student summary <span class="badge">{len(summary)} rows</span></h3>
  <p class="note">Per-student: age, family id, predicted superpop/population, ancestry proportions.</p>
  <div class="table-scroll">
    {table_html}
  </div>
</section>

<section id="downloads">
  <h3>Downloads</h3>
  <p class="note">Tables (TSV) for further analysis:</p>
  <div class="downloads">
    <a href="data/student_summary.tsv" download>student_summary.tsv</a>
    <a href="data/population_predictions.tsv" download>population_predictions.tsv</a>
    <a href="data/superpop_predictions.tsv" download>superpop_predictions.tsv</a>
    <a href="data/ancestry_proportions.tsv" download>ancestry_proportions.tsv</a>
    <a href="data/kinship_pairs.tsv" download>kinship_pairs.tsv</a>
    <a href="data/pca_coords.tsv" download>pca_coords.tsv</a>
    <a href="REPORT.md" download>REPORT.md</a>
  </div>
</section>

<footer>
  Pipeline: <code>plink2</code> (QC, LD-prune, PCA, KING) · Python (<code>sklearn</code>,
  <code>scipy NNLS</code>, <code>networkx</code>, <code>umap-learn</code>, <code>plotly</code>).
  All artefacts in <code>results/</code>; site source in <code>site/</code>.
</footer>

</main>
</body></html>"""

(SITE / "index.html").write_text(html, encoding="utf-8")

# netlify.toml
(SITE / "netlify.toml").write_text("""# Netlify config — pure static site, no build needed
[build]
  publish = "."

[[headers]]
  for = "/assets/*"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"

[[headers]]
  for = "/data/*"
  [headers.values]
    Cache-Control = "public, max-age=86400"

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
""", encoding="utf-8")

# _headers (alternative to netlify.toml headers)
(SITE / "_headers").write_text("""/assets/*
  Cache-Control: public, max-age=31536000, immutable

/data/*
  Cache-Control: public, max-age=86400
""", encoding="utf-8")

# Simple README for site
(SITE / "README.md").write_text("""# Digital Genealogist — Netlify site

Drag this folder onto https://app.netlify.com/drop to deploy.

Or with the Netlify CLI:

```
npm install -g netlify-cli
cd site
netlify deploy --prod
```

Pure static site — no build step required.
""", encoding="utf-8")

# Compute total size
total = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
print(f"Site built at {SITE}")
print(f"  index.html       {(SITE/'index.html').stat().st_size/1024:.0f} KB")
print(f"  assets/          {sum(p.stat().st_size for p in (SITE/'assets').rglob('*') if p.is_file())/1024:.0f} KB")
print(f"  data/            {sum(p.stat().st_size for p in (SITE/'data').rglob('*') if p.is_file())/1024:.0f} KB")
print(f"  total            {total/1024/1024:.1f} MB")
print(f"\nDeploy: drag the {SITE.name}/ folder onto https://app.netlify.com/drop")
