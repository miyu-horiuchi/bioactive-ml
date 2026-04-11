# Coding Agent Session: bioactive-ml

## Project
**Meal Shield** — Multi-task Graph Neural Network for predicting food peptide bioactivity against digestive enzyme targets. Combines ESM-2 protein language model embeddings with Graph Attention Networks and Topological Data Analysis.

## What was built in this session

### Data Pipeline (212 → thousands of peptides)
- **BIOPEP-UWM scraper** (`scrape_biopep.py`) — programmatic scraper for the BIOPEP-UWM bioactive peptide database. 19 activity categories, ~5,600 peptides. Two-phase approach: bulk search by activity, then detail page enrichment for IC50/EC50 values. File-based caching + rate limiting. Tested: 1,248 ACE inhibitory peptides confirmed.
- **DFBP + Peptipedia integration** (`fetch_external_dbs.py`) — scraper for DFBP (2,198 ACE peptides confirmed) and Peptipedia SQL dump parser (133K+ labeled peptides). Integrated into `fetch_peptides.py` via `--biopep` and `--dfbp` flags.
- **ChEMBL unit validation** — added `_convert_to_nM()` supporting nM/uM/mM/pM/M conversions. Previously only accepted nM and silently dropped everything else.

### Model Architecture Improvements
- **ESM-2 protein language model embeddings** (`esm_embeddings.py`) — per-residue 320-dim embeddings from `esm2_t6_8M_UR50D`, concatenated with 8-dim physicochemical features = 328-dim node features. Batch processing + disk caching.
- **Masked multi-task training** (`train_multitask()`) — trains all 6 targets simultaneously with masked loss. Each graph contributes loss only for its labeled target, enabling cross-task transfer learning.
- **Stratified k-fold cross-validation** (`cross_validate()`) — proper CV with mean ± std reporting per target.
- **Model defaults updated** — all 3 model classes (GNN, GNN+TDA, GIN) default to 6 targets including ACE and DPP-4 inhibitors.

### Training Results (first run)
```
ESM-2 + Multi-task, 100 epochs, 8,904 graphs, 391K parameters

Target                    R²      RMSE
alpha_glucosidase        0.800    0.566
lipase                   0.620    0.775
ace_inhibitor            0.614    0.982
dpp4_inhibitor           0.569    0.875
bile_acid_receptor       0.391    0.556
sodium_hydrogen_exchanger 0.376   0.941
```

### Performance Optimizations
- `torch.compile()` wrapping (~30% speedup)
- `num_workers=4` + `persistent_workers` on all DataLoaders (~2x data loading)
- GPU auto-detection: CUDA > MPS (PyTorch 2.4+) > CPU

### Visualization Suite (`visualize.py`)
- Prediction radar chart (multi-target profile per peptide)
- Dataset overview (4-panel: activity distribution, length, pIC50 boxplots, data sources)
- t-SNE embedding space (peptides colored by activity)
- Attention heatmap (per-residue importance across targets)
- Training metrics comparison (GNN vs GNN+TDA bar charts)
- CLI: `--dataset`, `--tsne`, `--radar SEQUENCE`, `--metrics`

### Infrastructure
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — tests on Python 3.9/3.10/3.11 + import lint
- **Dockerfile** + `.dockerignore` — CPU-only container for Modal/Railway/Fly
- **Test suite** — 52 tests across 4 files (data, model, train, baselines), 46 pass + 6 skip on macOS
- **`.env.example`** — documents all env vars
- **`pyproject.toml`** — added `fair-esm`, fixed `requires-python` to `>=3.9`, added `gdown` as optional

### Bug Fixes
- `evaluate.py` metrics key mismatch (`"r2"` vs `"R2"`) — would crash at runtime
- `train.py main()` wasn't calling `train_multitask()` or loading ESM cache — new features were dead code
- `server.py` hardcoded `feature_dim=11` — now reads from checkpoint metadata
- `json` import scoping bug in `train.py` — local import inside CV branch unbound global
- Orphaned `training.py` deleted (duplicated `train.py` functionality)

### Web Frontend
- Next.js frontend at `web/` with peptide input, prediction results, 3D molecule viewer (3Dmol.js), and attention-based explainability heatmap
- FastAPI backend at `server.py` with `/api/predict`, `/api/explain`, `/api/tda`, `/api/structure` endpoints

## Key Technical Decisions
1. **ESM-2 over hand-crafted features**: 8-dim physicochemical → 328-dim with ESM-2. Recent literature (ESMR4FBP 2024) shows ESM-2 outperforms classical descriptors by ~12% R² on peptide bioactivity.
2. **Masked multi-task loss over per-target training**: Enables cross-task transfer — every peptide trains all heads simultaneously. Critical for targets with sparse data.
3. **BIOPEP scraping over manual curation**: 212 curated peptides → infrastructure for thousands. Two-phase approach (bulk search → detail enrichment) with caching.
4. **torch.compile() over framework migration**: DGL would require full rewrite for marginal gains on small graphs. torch.compile gives 30% free.

### Round 3: Design Pipeline (from Hong et al. 2026 paper)
- 9 agents spawned in parallel to build the full design pipeline
- New modules: generate.py (MC/GA/enumeration), properties.py (7 developability scores), structure.py (ESMFold), pareto.py (multi-objective ranking), design.py (orchestrator), visualize_design.py
- API endpoints: /api/generate, /api/properties
- 36 new tests (all passing)
- DESIGN.md usage guide

### First Design Run Results (ACE inhibitor, genetic algorithm)
```
Rank  Sequence  pIC50   IC50(uM)  Solubility  Toxicity  Bitterness
1     KTRDK     5.578   2.64      1.0         0.0       0.0
2     VRSCR     5.561   2.75      1.0         0.1       0.04
3     KSRTR     5.576   2.66      1.0         0.2       0.0
4     PRRTK     5.589   2.58      1.0         0.2       0.0
5     VRDRK     5.569   2.70      1.0         0.0       0.0
```

### Round 4: Data Expansion
- BIOPEP-UWM scraped: 3,258 peptides (236 with IC50)
- DFBP scraped: 3,724 food-derived peptide sequences
- Merged dataset: 5,275 total peptides (25x expansion from 212)
- Retraining running with expanded data (11,740 graphs)

## Git Commits (this session)
```
6d81ab0 Add AI-guided peptide design pipeline
53da7b8 Add training performance optimizations
5d3b2b1 Wire new features into entry points + add visualizations
e3d5fd8 Add multi-task training, evaluation pipeline, API server, and web UI
```

## Git History
```
6d81ab0 Add AI-guided peptide design pipeline
53da7b8 Add training performance optimizations
5d3b2b1 Wire new features into entry points + add visualizations
e3d5fd8 Add multi-task training, evaluation pipeline, API server, and web UI
7d1196e Add expanded food peptide dataset (25 → 212 peptides)
f8400b1 Enhance TDA with cocycle features + DistPepFold integration
b08a428 Add persistent homology (TDA) features for GNN+topology model
5066889 Initial commit: multi-task GNN for food peptide bioactivity prediction
```
