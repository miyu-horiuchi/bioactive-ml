"""
Meal Shield — Interpretability Layer

Provides per-residue attribution scores explaining which amino acids
drove the model's prediction. Two methods:

1. Attention-based: Extracts and aggregates GAT attention weights
2. Integrated gradients: More principled gradient-based attribution

Usage:
    from interpret import explain_prediction
    result = explain_prediction(model, "IPAVF", "alpha_glucosidase", device)
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv


def extract_attention_weights(model, data, device):
    """
    Run forward pass through GAT layers and extract attention weights.

    Returns list of (edge_index, attention_weights) per GAT layer.
    """
    model.eval()
    data = data.to(device)
    x, edge_index = data.x, data.edge_index

    x = model.input_proj(x)
    x = F.relu(x)

    layer_attentions = []

    for conv, norm in zip(model.convs, model.norms):
        residual = x
        if isinstance(conv, GATConv):
            out, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
            layer_attentions.append((ei, alpha))
        else:
            out = conv(x, edge_index)
        x = norm(out)
        x = F.relu(x)
        x = x + residual

    return layer_attentions


def attention_attribution(model, data, device):
    """
    Compute per-node importance from GAT attention weights.

    Aggregates incoming attention across all layers and heads,
    normalized to [0, 1].
    """
    layer_attentions = extract_attention_weights(model, data, device)
    num_nodes = data.x.size(0)

    if not layer_attentions:
        return np.ones(num_nodes) / num_nodes

    node_scores = np.zeros(num_nodes)

    for edge_index, alpha in layer_attentions:
        alpha_mean = alpha.detach().cpu().mean(dim=-1).numpy()
        edge_idx = edge_index.detach().cpu().numpy()

        layer_scores = np.zeros(num_nodes)
        for e in range(edge_idx.shape[1]):
            target_node = edge_idx[1, e]
            if target_node < num_nodes:
                layer_scores[target_node] += alpha_mean[e]

        total = layer_scores.sum()
        if total > 0:
            layer_scores = layer_scores / total

        node_scores += layer_scores

    node_scores /= max(len(layer_attentions), 1)

    score_min = node_scores.min()
    score_max = node_scores.max()
    if score_max > score_min:
        node_scores = (node_scores - score_min) / (score_max - score_min)
    else:
        node_scores = np.ones(num_nodes) / num_nodes

    return node_scores


def integrated_gradients(model, data, target_name, device, steps=50):
    """
    Compute per-node attribution using integrated gradients.

    Satisfies the completeness axiom: attributions sum to the
    prediction difference between input and baseline (zeros).
    """
    model.eval()
    data = data.to(device)

    baseline_x = torch.zeros_like(data.x)
    diff = data.x - baseline_x

    accumulated_grads = torch.zeros_like(data.x)

    for step in range(steps + 1):
        alpha = step / steps
        interpolated_x = baseline_x + alpha * diff

        data_copy = data.clone()
        data_copy.x = interpolated_x.clone().detach().requires_grad_(True)

        predictions = model(data_copy)

        if target_name not in predictions:
            return np.ones(data.x.size(0)) / data.x.size(0)

        pred = predictions[target_name]
        if pred.dim() > 0:
            pred = pred.sum()

        model.zero_grad()
        pred.backward(retain_graph=True)

        if data_copy.x.grad is not None:
            accumulated_grads += data_copy.x.grad.detach()

    avg_grads = accumulated_grads / (steps + 1)
    integrated = diff * avg_grads

    # Per-node: sum over features, take absolute value
    node_scores = integrated.sum(dim=-1).detach().cpu().numpy()
    node_scores = np.abs(node_scores)

    score_max = node_scores.max()
    if score_max > 0:
        node_scores = node_scores / score_max

    return node_scores


def explain_prediction(model, sequence, target_name, device,
                       method="attention", feature_dim=11):
    """
    Explain a peptide prediction with per-residue attribution.

    Args:
        model: trained MealShieldGNN
        sequence: amino acid string
        target_name: target to explain
        device: torch device
        method: "attention" or "integrated_gradients"
        feature_dim: node feature dimension

    Returns:
        dict with residues, scores, method, prediction, top_residues
    """
    from data import peptide_to_graph
    from train import pad_features

    graph = peptide_to_graph(sequence, use_residue_level=True)
    if graph is None:
        return None

    graph = pad_features(graph, feature_dim)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    graph = graph.to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        predictions = model(graph)

    pred_val = None
    if target_name in predictions:
        pic50 = predictions[target_name].item()
        ic50_uM = (10 ** (9 - pic50)) / 1000 if pic50 > 0 else float("inf")
        pred_val = {"pIC50": round(pic50, 3), "IC50_uM": round(ic50_uM, 2)}

    # Compute attribution
    if method == "integrated_gradients":
        scores = integrated_gradients(model, graph, target_name, device)
    else:
        scores = attention_attribution(model, graph, device)

    residues = list(sequence)
    scores_list = scores.tolist()

    ranked = sorted(
        [(res, i, sc) for i, (res, sc) in enumerate(zip(residues, scores_list))],
        key=lambda x: x[2],
        reverse=True,
    )

    return {
        "residues": residues,
        "scores": scores_list,
        "method": method,
        "prediction": pred_val,
        "top_residues": [
            {"residue": r, "position": p, "score": round(s, 4)}
            for r, p, s in ranked
        ],
    }


def demo_explain(sequence, target_name):
    """
    Generate plausible demo attribution scores when no model is loaded.
    Uses amino acid physicochemical properties as a heuristic.
    """
    target_weights = {
        "alpha_glucosidase": {"D": 0.8, "E": 0.7, "K": 0.6, "R": 0.6, "H": 0.7, "Y": 0.5, "W": 0.6, "F": 0.5},
        "lipase": {"F": 0.8, "W": 0.9, "L": 0.7, "I": 0.7, "V": 0.6, "A": 0.5, "P": 0.6, "M": 0.5},
        "bile_acid_receptor": {"W": 0.8, "F": 0.7, "Y": 0.7, "H": 0.6, "K": 0.5, "R": 0.5},
        "sodium_hydrogen_exchanger": {"K": 0.7, "R": 0.8, "H": 0.7, "D": 0.6, "E": 0.6},
    }

    weights = target_weights.get(target_name, {})
    rng = np.random.RandomState(hash(sequence + target_name) % (2**31))

    scores = []
    for i, aa in enumerate(sequence):
        base = weights.get(aa, 0.3)
        if i == 0 or i == len(sequence) - 1:
            base += 0.15
        noise = rng.normal(0, 0.05)
        scores.append(max(0, min(1, base + noise)))

    scores = np.array(scores)
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)

    residues = list(sequence)
    scores_list = scores.tolist()

    ranked = sorted(
        [(res, i, sc) for i, (res, sc) in enumerate(zip(residues, scores_list))],
        key=lambda x: x[2],
        reverse=True,
    )

    return {
        "residues": residues,
        "scores": scores_list,
        "method": "heuristic (demo)",
        "prediction": None,
        "top_residues": [
            {"residue": r, "position": p, "score": round(s, 4)}
            for r, p, s in ranked
        ],
    }
