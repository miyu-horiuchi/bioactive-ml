"""
Meal Shield GNN — Visualization
Generates molecular graph visualizations like the reference image.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from data import smiles_to_graph, peptide_to_graph, AA_FEATURES


def graph_to_networkx(data):
    """Convert PyG Data to NetworkX graph."""
    G = nx.Graph()

    for i in range(data.x.shape[0]):
        G.add_node(i, features=data.x[i].tolist())

    edge_index = data.edge_index.t().tolist()
    seen = set()
    for src, dst in edge_index:
        edge = tuple(sorted([src, dst]))
        if edge not in seen:
            G.add_edge(src, dst)
            seen.add(edge)

    return G


def visualize_molecular_graph(data, title="Molecular Graph", save_path=None,
                                figsize=(10, 10), node_size=200, style="dark"):
    """
    Visualize a molecular graph in the style of the reference image.

    Args:
        data: PyG Data object
        title: Plot title
        save_path: Path to save the image
        figsize: Figure size
        node_size: Base node size
        style: 'dark' for dark nodes on white, 'light' for colored nodes
    """
    G = graph_to_networkx(data)

    # Layout
    if len(G.nodes) < 50:
        pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')

    # Node colors based on features
    features = data.x.numpy()

    if features.shape[1] >= 11:
        # Small molecule: color by atom type
        atom_nums = features[:, 0]
        color_map = {
            6: '#2C3E50',   # Carbon - dark blue-gray
            7: '#2980B9',   # Nitrogen - blue
            8: '#E74C3C',   # Oxygen - red
            16: '#F1C40F',  # Sulfur - yellow
            15: '#E67E22',  # Phosphorus - orange
            17: '#27AE60',  # Chlorine - green
        }
        node_colors = [color_map.get(int(a), '#1a1a2e') for a in atom_nums]
        node_sizes = [node_size * (1 + 0.3 * features[i, 1]) for i in range(len(G.nodes))]
    else:
        # Peptide: color by hydrophobicity
        if features.shape[1] >= 2:
            hydrophobicity = features[:, 1]  # Kyte-Doolittle
            norm = plt.Normalize(vmin=-4.5, vmax=4.5)
            cmap = plt.cm.coolwarm
            node_colors = [cmap(norm(h)) for h in hydrophobicity]
        else:
            node_colors = ['#1a1a2e'] * len(G.nodes)
        node_sizes = [node_size] * len(G.nodes)

    # Draw edges first (lighter)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#b0c4de',
        width=0.8,
        alpha=0.4,
        style='solid',
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors='#34495e',
        linewidths=0.5,
        alpha=0.9,
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    plt.close()

    return fig


def visualize_peptide_graph(sequence, save_path=None, figsize=(12, 8)):
    """
    Visualize a peptide as a residue-level graph with annotations.
    """
    data = peptide_to_graph(sequence, use_residue_level=True)
    if data is None:
        print(f"Could not create graph for {sequence}")
        return

    G = graph_to_networkx(data)

    # Layout: use spring with stronger repulsion for clarity
    pos = nx.spring_layout(G, k=3.0, iterations=200, seed=42)

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')

    # Color by property
    features = data.x.numpy()
    hydrophobicity = features[:, 1]

    norm = plt.Normalize(vmin=-4.5, vmax=4.5)
    cmap = plt.cm.RdYlBu_r  # Red=hydrophobic, Blue=hydrophilic

    node_colors = [cmap(norm(h)) for h in hydrophobicity]

    # Edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#b0c4de',
        width=1.5,
        alpha=0.5,
    )

    # Nodes
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=800,
        edgecolors='#2c3e50',
        linewidths=1.5,
    )

    # Labels (amino acid letters)
    labels = {i: aa for i, aa in enumerate(sequence)}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=12,
        font_weight='bold',
        font_color='white',
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Hydrophobicity (Kyte-Doolittle)', fontsize=10)

    # Annotations
    charge = sum(features[:, 2])
    avg_hydro = np.mean(hydrophobicity)
    mw = sum(features[:, 0])

    info_text = (
        f"Sequence: {sequence}\n"
        f"Length: {len(sequence)} residues\n"
        f"MW: {mw:.0f} Da\n"
        f"Net charge (pH 7): {charge:+.0f}\n"
        f"Avg hydrophobicity: {avg_hydro:.1f}"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontfamily='monospace')

    ax.set_title(f"Peptide Graph: {sequence}", fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    plt.close()

    return fig


def visualize_known_drugs(save_dir="figures"):
    """Visualize molecular graphs of known meal-shield drugs for comparison."""
    os.makedirs(save_dir, exist_ok=True)

    known_drugs = {
        "Acarbose (alpha-glucosidase inhibitor)": "OC1C(OC2(CO)OC(OC3OC(CO)C(O)C(O)C3N)C(O)C2O)C(O)C(O)C1O",
        "Orlistat (lipase inhibitor)": "CCCCCCCCCCC(CC1OC(=O)C1CCCCCC)OC(=O)C(CC(C)C)NC=O",
        "Ezetimibe (cholesterol absorber)": "OC1CCC(CC1)C(C1CCC(F)CC1)C1C(=O)OC(C1O)C1CCC(O)CC1",
    }

    for name, smiles in known_drugs.items():
        data = smiles_to_graph(smiles)
        if data is not None:
            safe_name = name.split("(")[0].strip().lower().replace(" ", "_")
            visualize_molecular_graph(
                data,
                title=name,
                save_path=os.path.join(save_dir, f"drug_{safe_name}.png"),
                figsize=(10, 10),
                node_size=300,
            )


def visualize_food_peptides(save_dir="figures"):
    """Visualize graphs of known food bioactive peptides."""
    os.makedirs(save_dir, exist_ok=True)

    peptides = [
        ("IPP", "ACE inhibitor (Calpis)"),
        ("VPP", "ACE inhibitor (Calpis)"),
        ("IPAVF", "Alpha-glucosidase inhibitor (bean)"),
        ("KLPGF", "Alpha-glucosidase inhibitor (silk)"),
        ("LPYPY", "DPP-4 inhibitor (Gouda cheese)"),
        ("LKPNM", "ACE inhibitor (bonito)"),
        ("PAGNFLPP", "Lipase inhibitor (soybean)"),
        ("LRSELAAWSR", "Alpha-glucosidase inhibitor (rice)"),
    ]

    for seq, desc in peptides:
        visualize_peptide_graph(
            seq,
            save_path=os.path.join(save_dir, f"peptide_{seq}.png"),
            figsize=(12, 8),
        )


def visualize_comparison_panel(save_dir="figures"):
    """Create a comparison panel of different peptide types."""
    os.makedirs(save_dir, exist_ok=True)

    categories = {
        "Sugar Blockers": [
            ("IPAVF", "bean"),
            ("KLPGF", "silk"),
            ("AKSPLF", "wheat"),
        ],
        "Fat Blockers": [
            ("PAGNFLPP", "soybean"),
            ("GPVRGPFPIIV", "casein"),
            ("VFPS", "tuna"),
        ],
        "Blood Pressure": [
            ("IPP", "Calpis milk"),
            ("VPP", "Calpis milk"),
            ("VY", "sardine"),
        ],
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 18), facecolor='white')

    for row_idx, (category, peptides) in enumerate(categories.items()):
        for col_idx, (seq, source) in enumerate(peptides):
            ax = axes[row_idx][col_idx]

            data = peptide_to_graph(seq, use_residue_level=True)
            if data is None:
                continue

            G = graph_to_networkx(data)
            pos = nx.spring_layout(G, k=3.0, iterations=200, seed=42)

            features = data.x.numpy()
            hydrophobicity = features[:, 1]
            norm = plt.Normalize(vmin=-4.5, vmax=4.5)
            cmap = plt.cm.RdYlBu_r
            node_colors = [cmap(norm(h)) for h in hydrophobicity]

            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#b0c4de', width=1.2, alpha=0.5)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=600,
                                   edgecolors='#2c3e50', linewidths=1.2)

            labels = {i: aa for i, aa in enumerate(seq)}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=11,
                                     font_weight='bold', font_color='white')

            ax.set_title(f"{seq}\n({source})", fontsize=11, fontweight='bold')
            ax.axis('off')

            if col_idx == 0:
                ax.text(-0.15, 0.5, category, transform=ax.transAxes,
                        fontsize=13, fontweight='bold', rotation=90,
                        verticalalignment='center', horizontalalignment='center')

    plt.suptitle("Meal Shield: Food Peptide Graphs by Function",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "peptide_comparison_panel.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_dir}/peptide_comparison_panel.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("MEAL SHIELD GNN -- Visualization")
    print("=" * 60)

    os.makedirs("figures", exist_ok=True)

    print("\n[1/3] Visualizing known drugs...")
    visualize_known_drugs()

    print("\n[2/3] Visualizing food peptides...")
    visualize_food_peptides()

    print("\n[3/3] Creating comparison panel...")
    visualize_comparison_panel()

    print("\nAll visualizations saved to figures/")
