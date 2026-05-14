"""Build family graphs from KING kinship and visualize.

KING-robust kinship thresholds (Manichaikul et al. 2010):
  > 0.354    duplicate / MZ twins
  0.177-0.354  1st degree (parent-child, full siblings)
  0.0884-0.177 2nd degree (half-siblings, grandparent, aunt/uncle)
  0.0442-0.0884 3rd degree (cousins)
  < 0.0442 unrelated

IBS0 helps distinguish parent-child (IBS0 ~ 0) from full siblings (IBS0 > 0.005):
parent-child must share at least one allele at every locus.
"""
from pathlib import Path
import math
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path("C:/Users/psgpe/Downloads/families_plus_popref.vcf")
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# Load KING table
kin = pd.read_csv(ROOT / "work/king.kin0", sep="\t")
kin = kin.rename(columns={"#IID1": "ID1", "IID1": "ID1", "IID2": "ID2"})
print(f"Loaded {len(kin)} related pairs (kinship>=0.0442).")

# Load metadata
fam = pd.read_csv(ROOT / "student_family_metadata.tsv", sep="\t")
age_map = dict(zip(fam["sample_id"], fam["age"]))
pred = pd.read_csv(RES / "population_predictions.tsv", sep="\t")
pop_map = dict(zip(pred["sample_id"], pred["pred_pop1"]))
super_map = dict(zip(pred["sample_id"], pred["pred_superpop"]))

# Classify relationship
def classify(row):
    k = row["KINSHIP"]
    ibs0 = row["IBS0"]
    if k > 0.354:
        return "MZ/dup"
    if k > 0.177:
        # 1st degree: parent-child vs full sibling by IBS0
        return "parent-child" if ibs0 < 0.005 else "full-sibling"
    if k > 0.0884:
        return "2nd-degree"
    if k > 0.0442:
        return "3rd-degree"
    return "weak"

kin["relation"] = kin.apply(classify, axis=1)
print("Relations:")
print(kin["relation"].value_counts())

# Filter to only student-student pairs (we care about student families)
students = set(fam["sample_id"])
kin_ss = kin[kin["ID1"].isin(students) & kin["ID2"].isin(students)].copy()
print(f"Student-student pairs: {len(kin_ss)} / {len(kin)}")

# Build graph: all related pairs (1st-3rd degree) among students
G = nx.Graph()
for sid in students:
    G.add_node(sid, age=age_map.get(sid, np.nan),
               pop=pop_map.get(sid, "?"), superpop=super_map.get(sid, "?"))
for _, r in kin_ss.iterrows():
    G.add_edge(r["ID1"], r["ID2"], kinship=r["KINSHIP"],
               ibs0=r["IBS0"], relation=r["relation"])

# Connected components = families (using 1st-2nd degree edges for family grouping)
strong = G.copy()
for u, v, d in list(G.edges(data=True)):
    if d["kinship"] < 0.0884:  # drop 3rd-degree from family-finding
        strong.remove_edge(u, v)
families = [sorted(c) for c in nx.connected_components(strong) if len(c) > 1]
families.sort(key=lambda c: -len(c))
print(f"\nFound {len(families)} families (size>=2 via 1st/2nd degree)")
for i, f in enumerate(families):
    ages = [age_map.get(s, "?") for s in f]
    print(f"  Family {i+1} (n={len(f)}): {f}  ages={ages}")

# Save kinship table with annotations
kin_ss["age1"] = kin_ss["ID1"].map(age_map)
kin_ss["age2"] = kin_ss["ID2"].map(age_map)
kin_ss["pop1"] = kin_ss["ID1"].map(pop_map)
kin_ss["pop2"] = kin_ss["ID2"].map(pop_map)
kin_ss.to_csv(RES / "kinship_pairs.tsv", sep="\t", index=False, float_format="%.4f")
print(f"Saved {RES/'kinship_pairs.tsv'}")

# --- Visualize each family ---
REL_COLORS = {"MZ/dup": "#7f0000", "parent-child": "#1f78b4",
              "full-sibling": "#33a02c", "2nd-degree": "#ff7f00",
              "3rd-degree": "#999999"}
REL_STYLES = {"MZ/dup": "-", "parent-child": "-", "full-sibling": "-",
              "2nd-degree": "--", "3rd-degree": ":"}

def plot_family(fam_nodes, ax, title):
    sub = G.subgraph(fam_nodes).copy()
    # also include 3rd-degree edges visible
    pos = nx.spring_layout(sub, seed=7, k=1.4 / math.sqrt(max(len(sub), 1)))
    # node sizes by age (children smaller, adults larger)
    sizes = []
    labels = {}
    for n in sub.nodes:
        a = sub.nodes[n].get("age")
        sizes.append(800 + (a if isinstance(a, (int, float)) and not np.isnan(a) else 30) * 12)
        pop = sub.nodes[n].get("pop", "?")
        a_lbl = f"\nage {int(a)}" if isinstance(a, (int, float)) and not np.isnan(a) else ""
        labels[n] = f"{n}{a_lbl}\n{pop}"
    # color by superpop
    SUPERPOP_COLORS = {
        "AFR": "#e41a1c", "EUR": "#377eb8", "EAS": "#4daf4a",
        "SAS": "#984ea3", "AMR": "#ff7f00", "STUDENT": "#222"}
    node_colors = [SUPERPOP_COLORS.get(sub.nodes[n].get("superpop", "?"), "#999") for n in sub.nodes]
    nx.draw_networkx_nodes(sub, pos, node_size=sizes, node_color=node_colors,
                           edgecolors="black", linewidths=1.2, alpha=0.85, ax=ax)
    # draw each relation type as separate edge style
    for rel, style in REL_STYLES.items():
        edges = [(u, v) for u, v, d in sub.edges(data=True) if d["relation"] == rel]
        if not edges:
            continue
        nx.draw_networkx_edges(sub, pos, edgelist=edges, edge_color=REL_COLORS[rel],
                               width=2.5 if rel == "parent-child" else 1.8, style=style,
                               alpha=0.85, ax=ax)
    nx.draw_networkx_labels(sub, pos, labels, font_size=8, ax=ax)
    # edge labels (kinship)
    elabels = {(u, v): f"{d['kinship']:.2f}" for u, v, d in sub.edges(data=True)}
    nx.draw_networkx_edge_labels(sub, pos, edge_labels=elabels, font_size=6, ax=ax,
                                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    ax.set_title(title, fontsize=11)
    ax.axis("off")

# Plot all families in a grid
n_fams = len(families)
if n_fams:
    n_cols = min(3, n_fams)
    n_rows = math.ceil(n_fams / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5.5 * n_rows), squeeze=False)
    for i, f in enumerate(families):
        ax = axes[i // n_cols][i % n_cols]
        plot_family(f, ax, f"Family {i+1} (n={len(f)})")
    # Hide empty axes
    for j in range(n_fams, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    # Legend
    leg = []
    for rel, c in REL_COLORS.items():
        leg.append(mpatches.Patch(color=c, label=rel))
    fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.suptitle("Student families — relationships (kinship-based)", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(RES / "family_trees.png", dpi=140)
    plt.close(fig)
    print(f"Saved {RES/'family_trees.png'}")

# Also save the overall student kinship graph (network view)
fig, ax = plt.subplots(figsize=(14, 12))
sub = nx.Graph()
sub.add_nodes_from(G.nodes(data=True))
sub.add_edges_from([(u, v, d) for u, v, d in G.edges(data=True) if d["kinship"] >= 0.0884])
pos = nx.spring_layout(sub, seed=42, k=0.35)
SUPERPOP_COLORS = {"AFR": "#e41a1c", "EUR": "#377eb8", "EAS": "#4daf4a",
                   "SAS": "#984ea3", "AMR": "#ff7f00", "STUDENT": "#222"}
node_colors = [SUPERPOP_COLORS.get(sub.nodes[n].get("superpop", "?"), "#999") for n in sub.nodes]
sizes = [350 if sub.degree(n) == 0 else 700 for n in sub.nodes]
nx.draw_networkx_nodes(sub, pos, node_size=sizes, node_color=node_colors,
                       edgecolors="black", linewidths=0.8, alpha=0.85, ax=ax)
for rel, style in REL_STYLES.items():
    edges = [(u, v) for u, v, d in sub.edges(data=True) if d["relation"] == rel]
    if edges:
        nx.draw_networkx_edges(sub, pos, edgelist=edges, edge_color=REL_COLORS[rel],
                               width=2.0, style=style, alpha=0.8, ax=ax)
nx.draw_networkx_labels(sub, pos, font_size=7, ax=ax)
leg = [mpatches.Patch(color=c, label=rel) for rel, c in REL_COLORS.items()]
leg += [mpatches.Patch(color=c, label=s) for s, c in SUPERPOP_COLORS.items() if s != "STUDENT"]
ax.legend(handles=leg, loc="best", fontsize=9, framealpha=0.9, ncol=2)
ax.set_title("All students — kinship network (1st & 2nd degree edges)", fontsize=13)
ax.axis("off")
fig.tight_layout()
fig.savefig(RES / "kinship_network.png", dpi=140)
plt.close(fig)
print(f"Saved {RES/'kinship_network.png'}")
