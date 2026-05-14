"""Pedigree-style family trees with ancestry pies as nodes.

Layout strategy per family:
  - Y axis = age (older at top). For visual clarity, we bin ages into
    'generations' so co-aged individuals sit on the same row but offset slightly.
  - X positions chosen to minimise edge crossings (recursive layout).
  - Each individual rendered as a pie chart of 5-way ancestry (AFR/AMR/EAS/EUR/SAS).
  - Edges:
      * parent-child: solid blue, vertical when generations differ
      * full-sibling: solid green, horizontal connector
      * MZ/duplicate: thick red double-line
      * 2nd-degree: dashed amber
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, FancyBboxPatch
from matplotlib.lines import Line2D
import networkx as nx

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
RES = ROOT / "results"
(RES / "families").mkdir(exist_ok=True)

summary = pd.read_csv(RES / "student_summary.tsv", sep="\t")
kin = pd.read_csv(RES / "kinship_pairs.tsv", sep="\t")

SP = ["AFR", "AMR", "EAS", "EUR", "SAS"]
SP_COLOR = {"AFR": "#e41a1c", "EUR": "#377eb8", "EAS": "#4daf4a",
            "SAS": "#984ea3", "AMR": "#ff7f00"}
REL_COLOR = {"MZ/dup": "#ff3030", "parent-child": "#4fc3f7",
             "full-sibling": "#7be57b", "2nd-degree": "#ffb74d",
             "3rd-degree": "#888"}

def draw_pie_node(ax, x, y, radius, ancestry, edge_color="white", label=None,
                  label_color="white", label_size=8):
    """Draw a pie chart at (x,y) showing ancestry split."""
    start = 90.0
    for sp in SP:
        v = ancestry.get(sp, 0.0)
        if v <= 0:
            continue
        end = start - 360 * v
        w = Wedge((x, y), radius, end, start, facecolor=SP_COLOR[sp],
                  edgecolor="none", linewidth=0, antialiased=True)
        ax.add_patch(w)
        start = end
    # outer ring
    ring = Circle((x, y), radius, fill=False, edgecolor=edge_color, linewidth=1.4)
    ax.add_patch(ring)
    if label:
        ax.text(x, y, label, ha="center", va="center", color=label_color,
                fontsize=label_size, fontweight="bold",
                path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground="#111")])

import matplotlib.patheffects

# Build family -> members
fam_id_to_members = summary.groupby("family_id")["sample_id"].apply(list).to_dict()

# Build relation graph (with edge data)
G_all = nx.Graph()
for sid in summary["sample_id"]:
    r = summary[summary["sample_id"] == sid].iloc[0]
    G_all.add_node(sid, age=r["age"], ancestry={sp: r[f"anc_{sp}"] for sp in SP},
                   pop=r["pred_pop1"], superpop=r["pred_superpop"])
for _, r in kin.iterrows():
    G_all.add_edge(r["ID1"], r["ID2"], rel=r["relation"], kin=r["KINSHIP"], ibs0=r["IBS0"])

def layout_family(members, G):
    """Layout: y = age (continuous), x = optimised to minimise edge crossings."""
    sub = G.subgraph(members).copy()
    # Cluster age-bands for x spacing only - members within 8 years jitter on same row
    ages = {m: sub.nodes[m]["age"] for m in members}
    sorted_m = sorted(members, key=lambda s: -ages[s])
    bands = []  # list of lists (age clusters)
    for m in sorted_m:
        if bands and ages[bands[-1][0]] - ages[m] <= 8:
            bands[-1].append(m)
        else:
            bands.append([m])
    # X within bands: try to put 1st-degree neighbors close together
    x_for = {}
    for band in bands:
        # initial: order by mean x of already-placed parents (older bands placed first)
        parents_known = {m: [n for n in sub.neighbors(m)
                              if sub.edges[m, n]["rel"] in {"parent-child", "MZ/dup", "full-sibling"}
                              and n in x_for] for m in band}
        for m in band:
            if parents_known[m]:
                x_for[m] = np.mean([x_for[n] for n in parents_known[m]])
            else:
                x_for[m] = 0.5
        # Spread within band to avoid overlap (sort by x, assign evenly)
        xs = sorted([(x_for[m], m) for m in band])
        for j, (_, m) in enumerate(xs):
            x_for[m] = (j + 1) / (len(band) + 1)
    # 2-3 passes of pulling siblings/spouses (full-sibling, MZ) close
    for _ in range(30):
        new_x = dict(x_for)
        for n in sub.nodes:
            siblings = [nb for nb in sub.neighbors(n)
                        if sub.edges[n, nb]["rel"] in {"full-sibling", "MZ/dup"}
                        and abs(ages[nb] - ages[n]) <= 10]
            children = [nb for nb in sub.neighbors(n)
                        if sub.edges[n, nb]["rel"] == "parent-child" and ages[nb] < ages[n]]
            parents = [nb for nb in sub.neighbors(n)
                        if sub.edges[n, nb]["rel"] == "parent-child" and ages[nb] > ages[n]]
            relevant = siblings + children + parents
            if relevant:
                avg = np.mean([x_for[r] for r in relevant])
                new_x[n] = 0.7 * x_for[n] + 0.3 * avg
        # Re-normalise within each band
        for band in bands:
            xs = sorted([(new_x[m], m) for m in band])
            for j, (_, m) in enumerate(xs):
                new_x[m] = (j + 1) / (len(band) + 1)
        x_for = new_x

    # y by age directly, normalised
    age_min = min(ages.values()); age_max = max(ages.values())
    span = max(age_max - age_min, 1)
    y_for = {m: (ages[m] - age_min) / span for m in members}
    pos = {m: (x_for[m], y_for[m]) for m in members}
    return pos, bands

def draw_family(family_id, members, ax, title_color="white"):
    pos, bands = layout_family(members, G_all)
    xs = np.array([pos[m][0] for m in members])
    ys = np.array([pos[m][1] for m in members])
    y_norm = {m: 0.15 + 0.7 * pos[m][1] for m in members}
    x_norm = {m: 0.1 + 0.8 * pos[m][0] for m in members}

    # To reduce clutter: if a 2nd-degree edge can be explained by a path through
    # 1st-degree (parent-child or full-sibling) relations of length 2, hide it.
    sub = G_all.subgraph(members).copy()
    first_deg = nx.Graph()
    first_deg.add_nodes_from(sub.nodes)
    for u, v, d in sub.edges(data=True):
        if d["rel"] in {"parent-child", "full-sibling", "MZ/dup"}:
            first_deg.add_edge(u, v)
    hidden_2nd = set()
    for u, v, d in sub.edges(data=True):
        if d["rel"] == "2nd-degree":
            try:
                if nx.shortest_path_length(first_deg, u, v) <= 2:
                    hidden_2nd.add((u, v))
            except nx.NetworkXNoPath:
                pass

    # Edges first (under nodes)
    for u, v, d in sub.edges(data=True):
        if (u, v) in hidden_2nd or (v, u) in hidden_2nd:
            continue
        x1, y1 = x_norm[u], y_norm[u]
        x2, y2 = x_norm[v], y_norm[v]
        rel = d["rel"]
        if rel == "MZ/dup":
            ax.plot([x1, x2], [y1, y2], color=REL_COLOR[rel], lw=5.0, alpha=0.95, zorder=1)
            ax.plot([x1, x2], [y1, y2], color="#0a0d12", lw=1.4, alpha=1, zorder=1.1)
        elif rel == "parent-child":
            ax.plot([x1, x2], [y1, y2], color=REL_COLOR[rel], lw=2.6, alpha=0.95, zorder=1)
        elif rel == "full-sibling":
            ax.plot([x1, x2], [y1, y2], color=REL_COLOR[rel], lw=2.2, alpha=0.95, zorder=1)
        elif rel == "2nd-degree":
            ax.plot([x1, x2], [y1, y2], color=REL_COLOR[rel], lw=1.3, alpha=0.7, zorder=1,
                    linestyle="--")
        elif rel == "3rd-degree":
            ax.plot([x1, x2], [y1, y2], color=REL_COLOR[rel], lw=0.9, alpha=0.5, zorder=1,
                    linestyle=":")

    # Nodes (pies)
    n = len(members)
    radius = max(0.045, 0.105 - 0.005 * n)
    for m in members:
        a = G_all.nodes[m]
        draw_pie_node(ax, x_norm[m], y_norm[m], radius, a["ancestry"],
                      edge_color="white", label=m, label_size=8 if n > 6 else 9)
        ax.text(x_norm[m], y_norm[m] - radius - 0.025,
                f"{int(a['age'])}y · {a['pop']}",
                ha="center", va="top", color="#aab2c2", fontsize=7)

    # Age axis labels on the left
    age_min = min(G_all.nodes[m]["age"] for m in members)
    age_max = max(G_all.nodes[m]["age"] for m in members)
    if age_max > age_min:
        for tick_age in np.linspace(age_min, age_max, 5):
            y = 0.15 + 0.7 * (tick_age - age_min) / (age_max - age_min)
            ax.text(0.05, y, f"{int(round(tick_age))}", ha="right", va="center",
                    color="#566273", fontsize=7)
        ax.plot([0.07, 0.07], [0.15, 0.85], color="#2a313e", lw=0.8)
        ax.text(0.05, 0.92, "age", ha="right", va="center", color="#566273", fontsize=8)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#0a0d12")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Family #{family_id}  ·  n={len(members)}", color=title_color,
                 fontsize=12, pad=10, fontweight="bold")

# --- Composite figure: all 15 families ---
fam_ids = sorted(fam_id_to_members.keys())
n_fams = len(fam_ids)
n_cols = 4
n_rows = math.ceil(n_fams / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.0 * n_rows),
                         facecolor="#0a0d12")
axes_flat = axes.flatten() if n_fams > 1 else [axes]
for i, fid in enumerate(fam_ids):
    members = fam_id_to_members[fid]
    draw_family(fid, members, axes_flat[i])
for j in range(n_fams, len(axes_flat)):
    axes_flat[j].axis("off")
    axes_flat[j].set_facecolor("#0a0d12")

# Big legend
fig.suptitle("STUDENT FAMILIES — ancestry pie nodes, kinship edges",
             color="white", fontsize=18, fontweight="light", y=0.995)
legend_elements = [
    Line2D([0],[0], color=REL_COLOR["MZ/dup"], lw=4, label="MZ / duplicate"),
    Line2D([0],[0], color=REL_COLOR["parent-child"], lw=2.6, label="parent–child"),
    Line2D([0],[0], color=REL_COLOR["full-sibling"], lw=2, label="full sibling"),
    Line2D([0],[0], color=REL_COLOR["2nd-degree"], lw=1.5, ls="--", label="2nd degree"),
]
sp_legend = [Line2D([0],[0], marker="o", color="w", markerfacecolor=SP_COLOR[s],
                    markersize=10, label=s, lw=0) for s in SP]
fig.legend(handles=legend_elements + sp_legend, loc="lower center",
           ncol=9, fontsize=10, frameon=False, labelcolor="white",
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.025, 1, 0.97])
out = RES / "family_pedigrees.png"
fig.savefig(out, dpi=170, facecolor="#0a0d12", bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}  ({out.stat().st_size/1024:.0f} KB)")

# --- Hi-res individual pedigrees for the 6 largest families ---
big = sorted(fam_id_to_members.items(), key=lambda x: -len(x[1]))[:6]
for fid, members in big:
    fig, ax = plt.subplots(figsize=(9, 8), facecolor="#0a0d12")
    draw_family(fid, members, ax)
    fig.legend(handles=legend_elements + sp_legend, loc="lower center",
               ncol=9, fontsize=10, frameon=False, labelcolor="white")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = RES / f"families/family_{fid:02d}.png"
    fig.savefig(out, dpi=170, facecolor="#0a0d12", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
