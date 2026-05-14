"""High-end summary visualizations:
  1. Radial family chart — all 70 students arranged on a circle, grouped by family,
     each as ancestry pie wedge, with kinship arcs inside.
  2. Sankey: predicted superpop → predicted population → family.
  3. Ancestry-galaxy heatmap: 70x5 ancestry matrix sorted by hierarchical clustering.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib.patches import Wedge, Circle, FancyArrowPatch, ConnectionPatch
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
import plotly.io as pio
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
RES = ROOT / "results"

summary = pd.read_csv(RES / "student_summary.tsv", sep="\t")
kin = pd.read_csv(RES / "kinship_pairs.tsv", sep="\t")

SP = ["AFR", "AMR", "EAS", "EUR", "SAS"]
SP_COLOR = {"AFR": "#e41a1c", "EUR": "#377eb8", "EAS": "#4daf4a",
            "SAS": "#984ea3", "AMR": "#ff7f00"}
REL_COLOR = {"MZ/dup": "#ff3030", "parent-child": "#4fc3f7",
             "full-sibling": "#7be57b", "2nd-degree": "#ffb74d"}

# ----- 1. Radial family chart -----
# Place students around a circle, grouped by family, sorted by dominant ancestry.
summary = summary.copy()
summary["dom"] = summary[[f"anc_{s}" for s in SP]].idxmax(axis=1).str.replace("anc_", "")
SP_ORDER = {"EUR": 0, "EAS": 1, "SAS": 2, "AMR": 3, "AFR": 4}
summary["dom_ord"] = summary["dom"].map(SP_ORDER)

# Order families by their dominant ancestry of the largest member, then size desc
fam_order_data = []
for fid, grp in summary.groupby("family_id"):
    dom_counts = grp["dom"].value_counts()
    fam_dom = dom_counts.index[0]
    fam_order_data.append((fid, fam_dom, len(grp), SP_ORDER[fam_dom]))
fam_order_data.sort(key=lambda t: (t[3], -t[2], t[0]))
fam_order = [t[0] for t in fam_order_data]

# Build ordered list of students with x positions (angles)
ordered_students = []
for fid in fam_order:
    members = summary[summary["family_id"] == fid].sort_values(
        ["dom_ord", "age"], ascending=[True, False])
    for _, r in members.iterrows():
        ordered_students.append({
            "sample_id": r["sample_id"], "family_id": fid,
            "ancestry": {s: r[f"anc_{s}"] for s in SP},
            "age": r["age"], "pop": r["pred_pop1"], "dom": r["dom"],
        })

n = len(ordered_students)
# Each student gets an angle slice. Insert a small gap between families.
gap_per_family = 0.02 * 2 * np.pi  # ~7 deg gap between family groups
total_gap = gap_per_family * len(fam_order)
slot = (2 * np.pi - total_gap) / n
angles = {}
a = np.pi / 2  # start at top
last_fam = None
for s in ordered_students:
    if last_fam is not None and s["family_id"] != last_fam:
        a -= gap_per_family
    angles[s["sample_id"]] = a
    a -= slot
    last_fam = s["family_id"]

# Plot
fig, ax = plt.subplots(figsize=(14, 14), facecolor="#0a0d12")
ax.set_facecolor("#0a0d12")
R_node = 1.0
R_arc = 0.78  # kinship arcs inside this radius
node_r = 0.05

# Family arcs (outer rings) and labels
from matplotlib.patches import Wedge as _Wedge
fam_color = {}
for fid in fam_order:
    members = [s for s in ordered_students if s["family_id"] == fid]
    angs = [angles[m["sample_id"]] for m in members]
    a_start = max(angs) + slot / 2
    a_end = min(angs) - slot / 2
    a_mid = (a_start + a_end) / 2
    fam_dom = max(SP, key=lambda s: np.mean([m["ancestry"][s] for m in members]))
    fam_color[fid] = SP_COLOR[fam_dom]
    # Outer arc band via Wedge (handles wraparound correctly)
    R1, R2 = R_node + 0.10, R_node + 0.15
    ax.add_patch(_Wedge((0, 0), R2, np.degrees(a_end), np.degrees(a_start),
                        width=R2 - R1, facecolor=fam_color[fid], alpha=0.55,
                        edgecolor="none", zorder=2))
    # Outer label
    R_lab = R_node + 0.23
    x_lab = R_lab * np.cos(a_mid); y_lab = R_lab * np.sin(a_mid)
    angle_deg = np.degrees(a_mid) % 360
    if angle_deg > 90 and angle_deg < 270:
        rot = angle_deg + 180; ha = "right"
    else:
        rot = angle_deg; ha = "left"
    ax.text(x_lab, y_lab, f"#{fid}  (n={len(members)})", color="white",
            ha=ha, va="center", fontsize=10, rotation=rot,
            rotation_mode="anchor", fontweight="bold")

# Kinship Bezier curves inside the circle
# Use parametric quadratic Bezier: P(t) = (1-t)^2 * P0 + 2(1-t)t * Pc + t^2 * P1
# Pc = origin (chord curve toward centre) — beautiful hub-style flow
R_chord_end = R_node - node_r - 0.005  # land just inside the node ring
for _, r in kin.iterrows():
    u, v = r["ID1"], r["ID2"]
    if u not in angles or v not in angles:
        continue
    a1, a2 = angles[u], angles[v]
    rel = r["relation"]
    color = REL_COLOR.get(rel, "#666")
    if rel == "MZ/dup":
        lw, alpha = 3.2, 0.95
    elif rel == "parent-child":
        lw, alpha = 1.5, 0.85
    elif rel == "full-sibling":
        lw, alpha = 1.3, 0.85
    elif rel == "2nd-degree":
        lw, alpha = 0.8, 0.5
    else:
        lw, alpha = 0.5, 0.3
    P0 = np.array([R_chord_end * np.cos(a1), R_chord_end * np.sin(a1)])
    P1 = np.array([R_chord_end * np.cos(a2), R_chord_end * np.sin(a2)])
    # Control point: pull toward origin (chord style); closer = more curve
    # If angles are close, less pull; if opposite, full pull through centre
    diff = abs((a1 - a2 + np.pi) % (2 * np.pi) - np.pi)  # 0..pi
    Pc = np.array([0.0, 0.0]) * 0  # exact centre
    # Better: pull to fraction of origin so curve bulges naturally
    pull = 0.1 + 0.4 * (1 - diff / np.pi)
    mid = (P0 + P1) / 2
    Pc = mid * pull  # less radial offset = more central curve
    t = np.linspace(0, 1, 80)
    B = ((1 - t)**2)[:, None] * P0 + (2 * (1 - t) * t)[:, None] * Pc + (t**2)[:, None] * P1
    ax.plot(B[:, 0], B[:, 1], color=color, lw=lw, alpha=alpha, zorder=1.5)

# Draw nodes (pie charts)
for s in ordered_students:
    a = angles[s["sample_id"]]
    x = R_node * np.cos(a); y = R_node * np.sin(a)
    # Pie
    start = 90.0
    for sp in SP:
        v = s["ancestry"][sp]
        if v <= 0:
            continue
        end = start - 360 * v
        ax.add_patch(Wedge((x, y), node_r, end, start,
                           facecolor=SP_COLOR[sp], edgecolor="none", zorder=3))
        start = end
    ax.add_patch(Circle((x, y), node_r, fill=False, edgecolor="white", lw=1, zorder=3.5))
    # Label outside
    R_lab = R_node + 0.045
    x_l = R_lab * np.cos(a); y_l = R_lab * np.sin(a)
    ax.text(x_l, y_l, s["sample_id"], color="white", ha="center", va="center",
            fontsize=6.5, zorder=4,
            rotation=np.degrees(a) - 90, rotation_mode="anchor",
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground="#0a0d12")])

# Centre title
ax.text(0, 0.05, "70 students", color="white", ha="center", va="center",
        fontsize=22, fontweight="light")
ax.text(0, -0.05, f"{len(fam_order)} families · 127 kin pairs",
        color="#aab2c2", ha="center", va="center", fontsize=12)

# Legends
from matplotlib.lines import Line2D
leg_rel = [Line2D([0],[0], color=c, lw=2.5, label=k) for k, c in REL_COLOR.items()]
leg_sp = [Line2D([0],[0], marker="o", markerfacecolor=SP_COLOR[s], markeredgecolor="white",
                 markersize=12, lw=0, label=s) for s in SP]
ax.legend(handles=leg_rel + leg_sp, loc="lower left", bbox_to_anchor=(0.02, 0.02),
          fontsize=9, ncol=2, frameon=False, labelcolor="white")

ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
out = RES / "radial_families.png"
fig.savefig(out, dpi=170, facecolor="#0a0d12", bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

# ----- 2. Sankey: superpop -> population -> family -----
sp_set = sorted(summary["pred_superpop"].unique(),
                key=lambda s: SP_ORDER.get(s, 99))
pop_set = list(summary.groupby(["pred_superpop", "pred_pop1"]).groups.keys())
pop_labels = [f"{p}" for (sp, p) in pop_set]
fam_set = fam_order
fam_labels = [f"#{f}" for f in fam_set]

# Nodes
labels = sp_set + pop_labels + fam_labels
node_colors = ([SP_COLOR[s] for s in sp_set]
               + [SP_COLOR[sp] for (sp, p) in pop_set]
               + [fam_color[f] for f in fam_set])

def idx_of(label, scope):
    if scope == "sp":
        return sp_set.index(label)
    if scope == "pop":
        return len(sp_set) + pop_set.index(label)
    if scope == "fam":
        return len(sp_set) + len(pop_set) + fam_set.index(label)

def hex_to_rgba(h, a):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

src, tgt, val, link_col = [], [], [], []
# sp -> pop
for (sp, pop), grp in summary.groupby(["pred_superpop", "pred_pop1"]):
    src.append(idx_of(sp, "sp"))
    tgt.append(idx_of((sp, pop), "pop"))
    val.append(len(grp))
    link_col.append(hex_to_rgba(SP_COLOR[sp], 0.65))
# pop -> fam
for (sp, pop, fid), grp in summary.groupby(["pred_superpop", "pred_pop1", "family_id"]):
    src.append(idx_of((sp, pop), "pop"))
    tgt.append(idx_of(fid, "fam"))
    val.append(len(grp))
    link_col.append(hex_to_rgba(SP_COLOR[sp], 0.55))

fig_sankey = go.Figure(go.Sankey(
    node=dict(label=labels, color=node_colors, pad=15, thickness=18,
              line=dict(color="#0a0d12", width=0.5)),
    link=dict(source=src, target=tgt, value=val, color=link_col),
))
fig_sankey.update_layout(
    title="<b>Sankey: Superpopulation → Population → Family</b>",
    paper_bgcolor="#0a0d12", font=dict(color="#eee", size=11),
    height=820, margin=dict(l=10, r=10, t=60, b=20),
)
fig_sankey.write_html(RES / "sankey.html", include_plotlyjs="cdn",
                      config={"displaylogo": False})
fig_sankey.write_image(RES / "sankey.png", width=1600, height=900, scale=2) if False else None
# Also write to a static png via kaleido if available
try:
    fig_sankey.write_image(RES / "sankey.png", width=1600, height=900, scale=2)
    print(f"Saved {RES/'sankey.png'}")
except Exception as e:
    print(f"(static sankey skipped: {e})")
print(f"Saved {RES/'sankey.html'}")

# ----- 3. Clustered ancestry heatmap -----
mat = summary[[f"anc_{s}" for s in SP]].values
# Hierarchical clustering on rows (students)
Z = hierarchy.linkage(pdist(mat, metric="euclidean"), method="ward")
order = hierarchy.leaves_list(Z)
mat_o = mat[order]
labels_o = summary["sample_id"].values[order]
fam_o = summary["family_id"].values[order]
sup_o = summary["pred_superpop"].values[order]

fig, ax = plt.subplots(figsize=(16, 8), facecolor="#0a0d12")
ax.set_facecolor("#0a0d12")
# Build as stacked bar
bottom = np.zeros(len(order))
for i, sp in enumerate(SP):
    ax.bar(range(len(order)), mat_o[:, i], bottom=bottom,
           color=SP_COLOR[sp], width=1.0, edgecolor="none", label=sp)
    bottom += mat_o[:, i]
# Family separators
prev = None
for i, f in enumerate(fam_o):
    if prev is not None and f != prev:
        ax.axvline(i - 0.5, color="white", lw=0.5, alpha=0.4)
    prev = f
# X tick labels
ax.set_xticks(range(len(order)))
ax.set_xticklabels(labels_o, rotation=90, fontsize=7, color="white")
# Color tick labels by family
fam_unique = list(dict.fromkeys(fam_o))
fam_color_map = {f: fam_color.get(f, "white") for f in fam_unique}
for i, tick in enumerate(ax.get_xticklabels()):
    tick.set_color(fam_color_map[fam_o[i]])
ax.set_xlim(-0.5, len(order) - 0.5)
ax.set_ylim(0, 1)
ax.set_ylabel("Ancestry proportion", color="white")
ax.tick_params(colors="white")
for s in ax.spines.values():
    s.set_color("#2a313e")
ax.set_title("Global ancestry — students ordered by hierarchical clustering "
             "(tick color = family)",
             color="white", fontsize=14, pad=12, fontweight="light")
ax.legend(loc="upper right", ncol=5, fontsize=10, frameon=False, labelcolor="white")
fig.tight_layout()
out = RES / "ancestry_clustered.png"
fig.savefig(out, dpi=150, facecolor="#0a0d12", bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
