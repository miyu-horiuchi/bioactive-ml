# Meal Shield

**Multi-task Graph Neural Network + Topological Data Analysis for predicting food peptide bioactivity.**

Meal Shield identifies natural food peptides that block sugar and fat digestion by predicting their inhibitory activity against six digestive enzyme targets. It combines Graph Attention Networks (GAT) for local molecular structure with Persistent Homology (TDA) for global molecular topology — rings, cavities, and binding pockets that GNNs alone cannot capture. Includes per-residue explainability via attention weights and integrated gradients.

## Why This Matters

Bioactive peptides in everyday foods (milk, fish, soy, rice) can naturally inhibit digestive enzymes — the same mechanism used by pharmaceutical drugs like acarbose and orlistat. Identifying these peptides computationally enables:

- **Functional food design** — formulate meals that modulate sugar/fat absorption
- **Drug discovery** — food-derived peptides as starting points for safer therapeutics
- **Personalized nutrition** — predict which peptides target which metabolic pathways

## Targets

| Target | ChEMBL ID | Mechanism |
|--------|-----------|-----------|
| Alpha-glucosidase | CHEMBL4979 | Blocks starch → glucose conversion |
| Pancreatic lipase | CHEMBL4822 | Blocks dietary fat digestion |
| FXR bile acid receptor | CHEMBL4829 | Modulates bile acid signaling |
| NHE3 exchanger | CHEMBL4145 | Regulates gut sodium absorption |
| ACE inhibitor | CHEMBL1808 | Angiotensin-converting enzyme (blood pressure) |
| DPP-4 inhibitor | CHEMBL284 | Dipeptidyl peptidase IV (blood sugar) |

## Architecture

```
Peptide sequence (e.g. "IPAVF")
        │
        ├──► GNN Branch (3× GAT layers, 4 attention heads)
        │    └── Atom/residue graph → local chemistry features (256d)
        │
        ├──► TDA Branch (Persistent Homology)
        │    └── 3D coords → Ripser → persistence statistics + cocycles (42d → 64d)
        │
        └──► Fusion → Shared trunk (128d) → 4 task-specific heads → pIC50 predictions
```

**GNN branch** learns local chemical neighborhoods — which atoms bond to what, aromatic rings, charge distributions.

**TDA branch** captures global shape — the size and persistence of molecular rings (H1), cavities and binding pockets (H2), and spatial extent of topological features via cocycle analysis.

## Dataset

- **ChEMBL**: Small-molecule inhibitors with measured IC50 values for each target
- **BIOPEP-UWM**: Thousands of bioactive peptides scraped programmatically (ACE inhibitors alone have 3,000+ entries)
- **Curated food peptides**: 212 hand-curated bioactive peptides from literature
- Expand the dataset: `python expand_dataset.py` (scrapes BIOPEP and merges with existing data)
- Source: `data/food_peptides.csv` (curated), `data/food_peptides_expanded.csv` (after expansion)

## Installation

```bash
# Clone
git clone https://github.com/miyuhoriuchi/bioactive-ml.git
cd bioactive-ml

# Install dependencies
pip install -e .

# For the web API server
pip install -e ".[server]"
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

## Training

### Train the base GNN model

```bash
python train.py
```

This will:
1. Download activity data from ChEMBL (cached to `data/chembl_combined.csv`)
2. Build molecular graphs for small molecules and peptides
3. Train per-target with early stopping
4. Save checkpoint to `checkpoints/meal_shield_gnn.pt`
5. Print evaluation metrics (R², RMSE) and sample predictions

### Train and compare GNN vs GNN+TDA

```bash
python train_tda.py
```

Trains both models side-by-side and reports the performance delta — this is the key result showing whether topological features improve predictions.

### Run full evaluation pipeline

```bash
python evaluate.py
```

Trains both models, saves checkpoints, and writes evaluation results to `RESULTS.md`.

## Inference

```python
from model import MealShieldGNN
from train import predict_peptide
import torch

device = torch.device("cpu")
model = MealShieldGNN(target_names=["alpha_glucosidase", "lipase", "bile_acid_receptor", "sodium_hydrogen_exchanger"])
model.load_state_dict(torch.load("checkpoints/meal_shield_gnn.pt", map_location=device))

results = predict_peptide(model, "IPAVF", model.target_names, device)
for target, pred in results.items():
    print(f"{target}: pIC50={pred['pIC50']}, IC50={pred['IC50_uM']} uM")
```

## Web Interface

The web interface consists of a FastAPI backend + Next.js frontend.

### Start the API server

```bash
uvicorn server:app --reload --port 8000
```

API endpoints:
- `GET /health` — server status and model info
- `POST /api/predict` — predict peptide bioactivity
- `POST /api/tda` — compute topological features
- `POST /api/explain` — per-residue attribution (attention weights or integrated gradients)
- `POST /api/structure` — generate 3D PDB structure for visualization
- `GET /api/targets` — list available prediction targets

### Start the frontend

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000 to use the interactive prediction interface.

### Deployment

- **Frontend**: Deploy `web/` to [Vercel](https://vercel.com) — set `BACKEND_URL` environment variable
- **Backend**: Deploy to [Modal](https://modal.com) or [Railway](https://railway.app) for GPU inference

## Explainability

The interpretability layer shows which amino acids drive each prediction:

```python
from interpret import explain_prediction
import torch

# After loading a trained model...
result = explain_prediction(model, "IPAVF", "lipase", device, method="attention")
for r in result["top_residues"][:3]:
    print(f"  {r['residue']} (pos {r['position']+1}): {r['score']:.1%} importance")
```

Two methods available:
- **Attention-based** (fast): Aggregates GAT attention weights across layers and heads
- **Integrated gradients** (principled): Path integral from zero baseline to actual input features

The web interface visualizes attributions as a heatmap on the 3D molecular structure — brighter/larger residues had more influence on the prediction.

## Expanding the Dataset

The BIOPEP-UWM scraper can pull thousands of bioactive peptides:

```bash
# Full scrape (all 9 activity categories, fetches IC50 from detail pages)
python expand_dataset.py

# Quick test (50 details per activity)
python expand_dataset.py --quick

# Specific activities only
python expand_dataset.py --activities ace_inhibitor dpp4_inhibitor
```

This merges BIOPEP data with the existing curated dataset, deduplicates, and outputs `data/food_peptides_expanded.csv`.

## Project Structure

```
bioactive-ml/
├── model.py             # GNN architectures (GAT, GAT+TDA, GIN)
├── data.py              # ChEMBL data pipeline + molecular graph construction
├── topology.py          # Persistent homology (TDA) feature computation
├── interpret.py         # Explainability (attention attribution, integrated gradients)
├── train.py             # Training script (multi-task learning)
├── train_tda.py         # GNN vs GNN+TDA comparison
├── evaluate.py          # Full evaluation pipeline with results output
├── expand_dataset.py    # BIOPEP-UWM scraper integration + dataset merging
├── scrape_biopep.py     # BIOPEP-UWM web scraper (17 activity categories)
├── fetch_peptides.py    # Curated food peptide dataset (212 peptides)
├── visualize.py         # Molecular graph visualization
├── server.py            # FastAPI backend (prediction, TDA, explain, 3D structure)
├── web/                 # Next.js frontend (3Dmol.js viewer, attribution heatmap)
├── data/
│   └── food_peptides.csv
├── checkpoints/         # Saved model weights (created by training)
└── pyproject.toml
```

## Citation

If you use this work, please cite:

```
Meal Shield: Multi-task GNN with Topological Data Analysis
for Food Peptide Bioactivity Prediction
```

## License

MIT
