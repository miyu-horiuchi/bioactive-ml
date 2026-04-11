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


def visualize_prediction_radar(predictions, sequence, save_path=None, figsize=(8, 8)):
    """
    Radar chart showing predicted pIC50 across all targets for one peptide.

    Args:
        predictions: dict from predict_peptide() — {target: {pIC50, IC50_uM}}
        sequence: peptide sequence string
        save_path: optional path to save
    """
    if not predictions:
        return

    targets = sorted(predictions.keys())
    values = [predictions[t]["pIC50"] for t in targets]

    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, len(targets), endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True), facecolor="white")

    ax.plot(angles_closed, values_closed, "o-", linewidth=2, color="#2980B9", markersize=6)
    ax.fill(angles_closed, values_closed, alpha=0.15, color="#2980B9")

    # Labels
    short_labels = [t.replace("_", "\n").replace("inhibitor", "inh.") for t in targets]
    ax.set_xticks(angles)
    ax.set_xticklabels(short_labels, fontsize=9)

    ax.set_ylim(0, max(values) * 1.2 if values else 10)
    ax.set_ylabel("pIC50", fontsize=10, labelpad=20)
    ax.set_title(f"Bioactivity Profile: {sequence}", fontsize=13, fontweight="bold", pad=25)

    # Annotate best target
    best_idx = np.argmax(values)
    best_target = targets[best_idx]
    best_ic50 = predictions[best_target]["IC50_uM"]
    ax.annotate(
        f"Best: {best_target}\nIC50={best_ic50:.1f} uM",
        xy=(angles[best_idx], values[best_idx]),
        xytext=(30, 20), textcoords="offset points",
        fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#E74C3C"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaa7", alpha=0.8),
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def visualize_dataset_overview(csv_path="data/food_peptides.csv", save_dir="figures"):
    """
    Four-panel overview of the dataset: activity distribution, length distribution,
    pIC50 distribution, and source breakdown.
    """
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")

    # 1. Activity distribution
    ax = axes[0, 0]
    counts = df["activity"].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white")
    ax.set_xlabel("Count")
    ax.set_title("Peptides by Activity", fontweight="bold")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)
    ax.invert_yaxis()

    # 2. Length distribution
    ax = axes[0, 1]
    lengths = df["length"] if "length" in df.columns else df["sequence"].str.len()
    ax.hist(lengths, bins=range(1, int(lengths.max()) + 2), color="#3498db",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Peptide Length (residues)")
    ax.set_ylabel("Count")
    ax.set_title("Length Distribution", fontweight="bold")
    ax.axvline(lengths.median(), color="#E74C3C", linestyle="--", label=f"Median={lengths.median():.0f}")
    ax.legend()

    # 3. pIC50 distribution by activity
    ax = axes[1, 0]
    active = df[df["activity"] != "inactive"]
    activities_sorted = active.groupby("activity")["pIC50"].median().sort_values(ascending=False).index
    boxplot_data = [active[active["activity"] == a]["pIC50"].dropna().values for a in activities_sorted]
    bp = ax.boxplot(boxplot_data, labels=[a.replace("_", "\n") for a in activities_sorted],
                    patch_artist=True, vert=True)
    for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(activities_sorted)))):
        patch.set_facecolor(color)
    ax.set_ylabel("pIC50")
    ax.set_title("Potency by Activity", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    # 4. Data source breakdown
    ax = axes[1, 1]
    if "data_source" in df.columns:
        source_counts = df["data_source"].value_counts()
    else:
        source_counts = df["source"].apply(
            lambda s: "BIOPEP" if "BIOPEP" in str(s) else
                      "DFBP" if "DFBP" in str(s) else "curated"
        ).value_counts()
    wedges, texts, autotexts = ax.pie(
        source_counts.values, labels=source_counts.index,
        autopct="%1.0f%%", startangle=90,
        colors=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"][:len(source_counts)],
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    ax.set_title("Data Sources", fontweight="bold")

    plt.suptitle("Meal Shield — Dataset Overview", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "dataset_overview.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {save_path}")
    plt.close()
    return fig


def visualize_attention_heatmap(model, sequence, target_names, device,
                                 feature_dim=11, save_path=None, figsize=(12, 4)):
    """
    Heatmap of per-residue attention scores across targets.

    Rows = targets, columns = residues. Shows which amino acids the model
    attends to for each bioactivity prediction.
    """
    from interpret import explain_prediction

    residues = list(sequence)
    scores_matrix = []
    valid_targets = []

    for target in target_names:
        result = explain_prediction(model, sequence, target, device,
                                    method="attention", feature_dim=feature_dim)
        if result and result["scores"]:
            scores_matrix.append(result["scores"])
            valid_targets.append(target)

    if not scores_matrix:
        print(f"  No attention scores for {sequence}")
        return

    mat = np.array(scores_matrix)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(residues)))
    ax.set_xticklabels(residues, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(valid_targets)))
    ax.set_yticklabels([t.replace("_", " ") for t in valid_targets], fontsize=9)

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] > 0.6 else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Attention Score")
    ax.set_title(f"Residue Attention: {sequence}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Residue")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def visualize_embedding_space(csv_path="data/food_peptides.csv", save_dir="figures",
                               figsize=(10, 8)):
    """
    t-SNE projection of peptide feature vectors, colored by activity.
    Uses physicochemical + composition features (no model needed).
    """
    import pandas as pd
    from sklearn.manifold import TSNE
    from baselines import peptide_features

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df[df["activity"] != "inactive"].copy()

    X = np.array([peptide_features(seq) for seq in df["sequence"]])

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    coords = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    activities = df["activity"].values
    unique_acts = sorted(set(activities))
    cmap = plt.cm.get_cmap("tab10", len(unique_acts))
    act_to_color = {a: cmap(i) for i, a in enumerate(unique_acts)}

    for act in unique_acts:
        mask = activities == act
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[act_to_color[act]], label=act.replace("_", " "),
                   s=50, alpha=0.7, edgecolors="white", linewidths=0.5)

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_title("Peptide Feature Space (t-SNE)", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "embedding_tsne.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {save_path}")
    plt.close()
    return fig


def visualize_training_metrics(metrics_path="checkpoints/comparison_results.json",
                                save_dir="figures"):
    """
    Bar chart comparing GNN vs GNN+TDA R² across targets.
    Reads from the JSON saved by evaluate.py.
    """
    import json

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(metrics_path):
        print(f"  No metrics file at {metrics_path} — run evaluate.py first")
        return

    with open(metrics_path) as f:
        data = json.load(f)

    gnn = data.get("gnn", {})
    tda = data.get("tda", {})
    targets = sorted(set(list(gnn.keys()) + list(tda.keys())))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    # R² comparison
    ax = axes[0]
    x = np.arange(len(targets))
    width = 0.35
    gnn_r2 = [gnn.get(t, {}).get("R2", 0) for t in targets]
    tda_r2 = [tda.get(t, {}).get("R2", 0) for t in targets]
    ax.bar(x - width / 2, gnn_r2, width, label="GNN", color="#3498db", edgecolor="white")
    ax.bar(x + width / 2, tda_r2, width, label="GNN+TDA", color="#e74c3c", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in targets], fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title("Prediction Accuracy (R²)", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1)

    # RMSE comparison
    ax = axes[1]
    gnn_rmse = [gnn.get(t, {}).get("RMSE", 0) for t in targets]
    tda_rmse = [tda.get(t, {}).get("RMSE", 0) for t in targets]
    ax.bar(x - width / 2, gnn_rmse, width, label="GNN", color="#3498db", edgecolor="white")
    ax.bar(x + width / 2, tda_rmse, width, label="GNN+TDA", color="#e74c3c", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in targets], fontsize=8)
    ax.set_ylabel("RMSE (pIC50)")
    ax.set_title("Prediction Error (RMSE)", fontweight="bold")
    ax.legend()

    plt.suptitle("Meal Shield — GNN vs GNN+TDA", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {save_path}")
    plt.close()
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Meal Shield Visualization")
    parser.add_argument("--all", action="store_true", help="Generate all visualizations")
    parser.add_argument("--graphs", action="store_true", help="Peptide/drug graphs")
    parser.add_argument("--dataset", action="store_true", help="Dataset overview")
    parser.add_argument("--tsne", action="store_true", help="t-SNE embedding space")
    parser.add_argument("--metrics", action="store_true", help="Training metrics comparison")
    parser.add_argument("--radar", type=str, default=None,
                        help="Radar chart for a specific peptide (requires trained model)")
    args = parser.parse_args()

    if not any([args.all, args.graphs, args.dataset, args.tsne, args.metrics, args.radar]):
        args.all = True

    print("=" * 60)
    print("MEAL SHIELD GNN -- Visualization")
    print("=" * 60)

    os.makedirs("figures", exist_ok=True)

    if args.all or args.graphs:
        print("\n[1/5] Visualizing known drugs...")
        visualize_known_drugs()
        print("\n[2/5] Visualizing food peptides...")
        visualize_food_peptides()
        print("\n[3/5] Creating comparison panel...")
        visualize_comparison_panel()

    if args.all or args.dataset:
        print("\n[4/5] Dataset overview...")
        visualize_dataset_overview()

    if args.all or args.tsne:
        print("\n[5/5] t-SNE embedding space...")
        visualize_embedding_space()

    if args.metrics:
        print("\nTraining metrics comparison...")
        visualize_training_metrics()

    if args.radar:
        print(f"\nRadar chart for {args.radar}...")
        # Needs a trained model — generate demo predictions
        from data import TARGETS
        demo_preds = {}
        for t in TARGETS:
            np.random.seed(hash(args.radar + t) % (2**31))
            pic50 = np.random.uniform(3.0, 6.5)
            demo_preds[t] = {"pIC50": round(pic50, 2), "IC50_uM": round(10**(6 - pic50), 1)}
        visualize_prediction_radar(
            demo_preds, args.radar,
            save_path=f"figures/radar_{args.radar}.png"
        )

    print("\nAll visualizations saved to figures/")
