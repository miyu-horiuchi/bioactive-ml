# Meal Shield Design Pipeline

AI-guided peptide design for food bioactivity. Generates candidate peptides,
scores them on safety and developability, applies Pareto filtering, and
optionally predicts 3D structure.

## Pipeline overview

```
generate  -->  score  -->  filter  -->  rank  -->  structure
(generate.py)  (properties.py)  (pareto.py)  (design.py)  (structure.py)
```

1. **Generate** candidate sequences targeting a specific enzyme (e.g. ACE inhibitor).
2. **Score** each candidate on six developability properties.
3. **Filter** using Pareto-optimal selection across activity + properties.
4. **Rank** final candidates by weighted objective.
5. **Structure** prediction via ESMFold for top hits.

## Quick start

```bash
# Install with design dependencies
pip install -e ".[design,server]"

# Generate ACE-inhibitor peptides (genetic algorithm, length 5)
python design.py --target ace_inhibitor --method genetic --length 5 --top-k 10

# Score a single peptide
python properties.py --sequence LIWKL

# Pareto filter a batch CSV
python pareto.py --input candidates.csv --objectives pIC50 solubility --output pareto_front.csv

# Predict structure for top hit
python structure.py --sequence LIWKL --output LIWKL.pdb
```

## Modules

### generate.py

Produces candidate peptide sequences optimized for a target enzyme.

| Method      | Flag          | Description                                      |
|-------------|---------------|--------------------------------------------------|
| Genetic     | `--method genetic` | Evolutionary search with crossover + mutation |
| Monte Carlo | `--method mc`      | Metropolis-Hastings sampling in sequence space |
| Enumerate   | `--method enumerate`| Exhaustive search (short peptides only, <=4 aa)|

Key arguments: `--target`, `--length`, `--n-candidates`, `--top-k`.

### properties.py

Scores a peptide on six developability properties. Each returns a float in [0, 1]
where higher is better (safer / more developable).

| Property       | What it measures                                    |
|----------------|-----------------------------------------------------|
| `toxicity`     | Predicted non-toxicity (1 = non-toxic)              |
| `hemolysis`    | Predicted non-hemolytic activity (1 = safe)         |
| `solubility`   | Aqueous solubility score                            |
| `permeability` | Membrane permeability (intestinal absorption)       |
| `stability`    | Resistance to proteolytic degradation               |
| `bitterness`   | Predicted non-bitterness (1 = not bitter)           |

### pareto.py

Multi-objective Pareto front extraction. Filters candidates to the
non-dominated set across any combination of objectives (e.g. pIC50 vs.
solubility vs. toxicity). Supports weighted ranking within the Pareto front.

### structure.py

3D structure prediction. Uses ESMFold (when available) or falls back to
RDKit distance geometry for short peptides. Outputs PDB format compatible
with 3Dmol.js and PyMOL.

### design.py

Orchestrator that chains generate -> score -> filter -> rank -> structure.
Single entry point for the full pipeline.

Key arguments:
- `--target`: enzyme target name (e.g. `ace_inhibitor`, `dpp4_inhibitor`)
- `--method`: generation strategy (`genetic`, `mc`, `enumerate`)
- `--length`: peptide length (2-15 residues)
- `--top-k`: number of final candidates to return
- `--predict-structure`: run ESMFold on top hits
- `--output`: output directory for results

## API endpoints

The server (`server.py`) exposes these endpoints for web integration:

| Method | Endpoint           | Description                          |
|--------|--------------------|--------------------------------------|
| POST   | `/api/generate`    | Generate candidate peptides          |
| POST   | `/api/properties`  | Score developability properties      |
| POST   | `/api/predict`     | Predict bioactivity (pIC50)          |
| POST   | `/api/structure`   | Generate 3D structure (PDB)          |
| POST   | `/api/explain`     | Residue-level attribution            |
| GET    | `/api/targets`     | List available enzyme targets        |
| GET    | `/health`          | Server health check                  |

Example request:

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"target": "ace_inhibitor", "method": "genetic", "length": 5, "top_k": 5}'
```

## Output format

The pipeline writes a JSON file per run:

```json
{
  "target": "ace_inhibitor",
  "method": "genetic",
  "candidates": [
    {
      "rank": 1,
      "sequence": "LIWKL",
      "pIC50": 6.82,
      "IC50_uM": 0.15,
      "properties": {
        "toxicity": 0.95,
        "hemolysis": 0.88,
        "solubility": 0.72,
        "permeability": 0.64,
        "stability": 0.81,
        "bitterness": 0.77
      },
      "pareto_optimal": true,
      "pdb_file": "structures/LIWKL.pdb"
    }
  ]
}
```

## References

- Hong, M. et al. (2026). "AI-Designed Peptides as Tools for Biochemistry."
  *Biochemistry*. Describes the generate-score-filter paradigm for
  computational peptide design applied to food-derived bioactive peptides.
