# Meal Shield -- AI-Guided Food Peptide Bioactivity Prediction and Design

Multi-task Graph Attention Network combining ESM-2 protein language model embeddings with topological data analysis to predict food peptide bioactivity against six digestive enzyme targets. Includes an AI-guided design pipeline (genetic algorithm / Monte Carlo) that generates novel peptide candidates, scores them on safety and developability properties, applies Pareto-optimal filtering, and predicts 3D structures via ESMFold. Trained on 15,000+ peptides aggregated from ChEMBL, BIOPEP-UWM, DFBP, and curated literature.

## Architecture

```
Data Sources          Model                    Applications
─────────────       ─────────────            ─────────────
ChEMBL (8,794)  →   ESM-2 (320-dim)     →   Bioactivity Prediction
BIOPEP (3,258)  →   + GAT (3-layer)     →   Peptide Design (GA/MC)
DFBP (3,724)    →   Multi-task heads    →   Property Scoring
Curated (212)   →   (6 targets)         →   3D Structure (ESMFold)
```

## Results

| Target | R² | RMSE | Test N |
|---|---|---|---|
| alpha_glucosidase | 0.831 | 0.633 | 275 |
| dpp4_inhibitor | 0.745 | 0.872 | 519 |
| ace_inhibitor | 0.676 | 0.735 | 671 |
| lipase | 0.580 | 0.859 | 408 |
| sodium_hydrogen_exchanger | 0.425 | 0.903 | 77 |
| bile_acid_receptor | 0.398 | 0.553 | 400 |

## Quick Start

```bash
pip install -e .
python fetch_peptides.py --biopep --dfbp    # Expand dataset
python train.py --esm --multitask           # Train
python design.py --target ace_inhibitor --method genetic --length 5  # Design
python -m uvicorn server:app --port 8000    # API
cd web && npm run dev                       # Frontend
```

## Project Structure

| File | Description |
|---|---|
| `data.py` | ChEMBL data pipeline and molecular graph construction |
| `model.py` | Multi-task GAT / GIN model architectures |
| `train.py` | Training loop with masked multi-task loss and cross-validation |
| `train_tda.py` | GNN + persistent homology comparison training |
| `evaluate.py` | Full evaluation pipeline with checkpoint saving and results export |
| `baselines.py` | Random Forest and Ridge Regression baselines using fingerprints |
| `esm_embeddings.py` | ESM-2 per-residue embeddings (320-dim) with disk caching |
| `topology.py` | Persistent homology features via Ripser (rings, cavities, tunnels) |
| `server.py` | FastAPI backend with prediction, explanation, and structure endpoints |
| `visualize.py` | Dataset overview, t-SNE, radar charts, attention heatmaps |
| `visualize_design.py` | Design pipeline plots: Pareto fronts, property radars, sequence logos |
| `generate.py` | Peptide sequence generator (Monte Carlo, genetic algorithm, enumeration) |
| `properties.py` | Developability scoring: toxicity, hemolysis, solubility, stability, bitterness |
| `pareto.py` | Multi-objective Pareto front extraction and diversity-aware ranking |
| `structure.py` | ESMFold 3D structure prediction with PDB output and pLDDT scoring |
| `design.py` | End-to-end design orchestrator (generate -> score -> filter -> rank -> structure) |
| `fetch_peptides.py` | Multi-source data collection pipeline (literature, BIOPEP, DFBP) |
| `scrape_biopep.py` | BIOPEP-UWM web scraper with two-phase enrichment and caching |
| `fetch_external_dbs.py` | DFBP and Peptipedia integration for external peptide databases |
| `interpret.py` | Per-residue attribution via GAT attention weights and integrated gradients |

## Design Pipeline

The design pipeline generates novel food peptides optimized for a target enzyme. It chains five stages: candidate generation, developability scoring, bioactivity prediction, Pareto-optimal selection, and 3D structure prediction.

```bash
python design.py --target ace_inhibitor --method genetic --length 5 --top-k 5
```

Sample output (ACE inhibitor, genetic algorithm, length 5):

| Rank | Sequence | pIC50 | IC50 (uM) | Solubility | Toxicity | Bitterness |
|---|---|---|---|---|---|---|
| 1 | KTRDK | 5.578 | 2.64 | 1.0 | 0.0 | 0.0 |
| 2 | VRSCR | 5.561 | 2.75 | 1.0 | 0.1 | 0.04 |
| 3 | KSRTR | 5.576 | 2.66 | 1.0 | 0.2 | 0.0 |
| 4 | PRRTK | 5.589 | 2.58 | 1.0 | 0.2 | 0.0 |
| 5 | VRDRK | 5.569 | 2.70 | 1.0 | 0.0 | 0.0 |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server health check |
| GET | `/api/targets` | List available enzyme targets |
| POST | `/api/predict` | Predict bioactivity (pIC50) for a peptide sequence |
| POST | `/api/explain` | Residue-level attribution scores |
| POST | `/api/tda` | Topological data analysis features |
| POST | `/api/structure` | 3D structure prediction (PDB format) |
| POST | `/api/properties` | Developability property scoring |
| POST | `/api/generate` | Generate candidate peptides for a target |

Start the server:

```bash
pip install -e ".[server]"
python -m uvicorn server:app --port 8000
```

## Web UI

The Next.js frontend lives in `web/` and provides an interactive interface for peptide analysis:

- Peptide sequence input with real-time bioactivity prediction across all six targets
- 3D molecule viewer powered by 3Dmol.js for visualizing predicted structures
- Attention heatmaps showing per-residue contributions to each prediction
- Design pipeline integration for generating and ranking novel candidates

```bash
cd web && npm install && npm run dev
```

## Data Sources

| Source | Peptides | Description |
|---|---|---|
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | 8,794 | Curated bioassay data with IC50/Ki values |
| [BIOPEP-UWM](https://biochemia.uwm.edu.pl/biopep-uwm/) | 3,258 | Bioactive peptides across 19 activity categories |
| [DFBP](http://www.omic.tech/dfbpapp/) | 3,724 | Food-derived bioactive peptides with IC50 values |
| Curated literature | 212 | Hand-curated peptides from published studies |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Citation

```bibtex
@article{hong2026ai,
  title={AI-Designed Peptides as Tools for Biochemistry},
  author={Hong, L. and Vincoff, S. and Chatterjee, P.},
  journal={Biochemistry},
  year={2026}
}
```

## License

MIT
