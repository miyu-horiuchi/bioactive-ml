"""
Meal Shield GNN -- Peptide Sequence Generator

Uses the trained multi-task GNN as a scoring oracle to propose novel peptides
optimized for a specific bioactivity target (e.g. ACE inhibition, DPP-4
inhibition, alpha-glucosidase inhibition).

Three generation strategies:
  1. Monte Carlo (simulated annealing) -- mutate one residue at a time
  2. Genetic algorithm -- population-based crossover + mutation
  3. Exhaustive enumeration -- brute-force all di/tri-peptides

Usage examples:
  python generate.py --target ace_inhibitor --method mc --length 5 --n-candidates 200 --top-k 10
  python generate.py --target dpp4_inhibitor --method ga --length 4 --n-candidates 300 --top-k 20
  python generate.py --target alpha_glucosidase --method exhaustive --length 3 --top-k 15
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from itertools import product
from typing import Dict, List, Optional, Tuple

import torch

from model import MealShieldGNN
from train import predict_peptide

# Standard amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def random_sequence(length: int) -> str:
    """Generate a random peptide sequence of the given length."""
    return "".join(random.choice(AMINO_ACIDS) for _ in range(length))


def mutate_single(sequence: str) -> str:
    """Mutate a single random residue to a different amino acid."""
    seq = list(sequence)
    pos = random.randint(0, len(seq) - 1)
    current = seq[pos]
    alternatives = [aa for aa in AMINO_ACIDS if aa != current]
    seq[pos] = random.choice(alternatives)
    return "".join(seq)


def crossover(parent_a: str, parent_b: str) -> Tuple[str, str]:
    """Single-point crossover between two same-length sequences."""
    assert len(parent_a) == len(parent_b), "Parents must have equal length"
    if len(parent_a) <= 1:
        return parent_a, parent_b
    point = random.randint(1, len(parent_a) - 1)
    child_a = parent_a[:point] + parent_b[point:]
    child_b = parent_b[:point] + parent_a[point:]
    return child_a, child_b


def score_peptide(
    model: MealShieldGNN,
    sequence: str,
    target: str,
    target_names: List[str],
    device: torch.device,
    feature_dim: int,
) -> Optional[float]:
    """Score a peptide and return its pIC50 for the given target.

    Returns None if the peptide cannot be featurized.
    """
    results = predict_peptide(
        model, sequence, target_names, device,
        feature_dim=feature_dim,
    )
    if results is None or target not in results:
        return None
    return results[target]["pIC50"]


def score_peptide_full(
    model: MealShieldGNN,
    sequence: str,
    target: str,
    target_names: List[str],
    device: torch.device,
    feature_dim: int,
) -> Optional[Dict]:
    """Score a peptide and return full result dict for the target.

    Returns None if the peptide cannot be featurized.
    """
    results = predict_peptide(
        model, sequence, target_names, device,
        feature_dim=feature_dim,
    )
    if results is None or target not in results:
        return None
    return results[target]


# ------------------------------------------------------------------
# Strategy 1: Monte Carlo (Simulated Annealing)
# ------------------------------------------------------------------

def generate_monte_carlo(
    model: MealShieldGNN,
    target: str,
    target_names: List[str],
    device: torch.device,
    feature_dim: int,
    length: int = 5,
    n_candidates: int = 100,
    n_restarts: int = 10,
    steps_per_restart: Optional[int] = None,
    t_start: float = 1.0,
    t_end: float = 0.01,
) -> List[Tuple[str, float]]:
    """Monte Carlo sampling with simulated annealing.

    Starts from a random sequence, mutates one residue at a time, and accepts
    the mutation if pIC50 improves.  Worse mutations are accepted with a
    Boltzmann probability that decreases according to the temperature schedule.

    Multiple independent restarts are used to explore sequence space more
    broadly.  The best ``n_candidates`` unique sequences seen across all
    restarts are returned.
    """
    if steps_per_restart is None:
        steps_per_restart = max(n_candidates // n_restarts, 50)

    seen: Dict[str, float] = {}

    for restart in range(n_restarts):
        current_seq = random_sequence(length)
        current_score = score_peptide(
            model, current_seq, target, target_names, device, feature_dim,
        )
        if current_score is None:
            continue
        seen[current_seq] = current_score

        for step in range(steps_per_restart):
            # Temperature schedule: linear decay
            progress = step / max(steps_per_restart - 1, 1)
            temperature = t_start + (t_end - t_start) * progress

            candidate_seq = mutate_single(current_seq)
            candidate_score = score_peptide(
                model, candidate_seq, target, target_names, device, feature_dim,
            )
            if candidate_score is None:
                continue

            seen[candidate_seq] = candidate_score

            delta = candidate_score - current_score  # higher pIC50 is better

            if delta > 0:
                # Always accept improvements
                current_seq = candidate_seq
                current_score = candidate_score
            else:
                # Accept worse solutions with Boltzmann probability
                acceptance_prob = math.exp(delta / max(temperature, 1e-10))
                if random.random() < acceptance_prob:
                    current_seq = candidate_seq
                    current_score = candidate_score

        if (restart + 1) % max(n_restarts // 5, 1) == 0:
            print(
                f"  MC restart {restart + 1}/{n_restarts} | "
                f"best so far: {max(seen.values()):.3f} pIC50 | "
                f"unique sequences: {len(seen)}"
            )

    # Sort by pIC50 descending, return top n_candidates
    ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    return ranked[:n_candidates]


# ------------------------------------------------------------------
# Strategy 2: Genetic Algorithm
# ------------------------------------------------------------------

def generate_genetic(
    model: MealShieldGNN,
    target: str,
    target_names: List[str],
    device: torch.device,
    feature_dim: int,
    length: int = 5,
    n_candidates: int = 100,
    population_size: int = 50,
    n_generations: int = 40,
    mutation_rate: float = 0.3,
    tournament_size: int = 3,
    elite_frac: float = 0.1,
) -> List[Tuple[str, float]]:
    """Genetic algorithm for peptide optimization.

    Maintains a population of sequences.  Each generation:
      1. Evaluate fitness (pIC50 for target)
      2. Select parents via tournament selection
      3. Crossover to produce offspring
      4. Mutate offspring
      5. Elite carry-over
    """
    # Initialize population with random sequences
    population: List[str] = [random_sequence(length) for _ in range(population_size)]

    all_seen: Dict[str, float] = {}

    def evaluate_population(pop: List[str]) -> List[Tuple[str, float]]:
        scored = []
        for seq in pop:
            if seq in all_seen:
                scored.append((seq, all_seen[seq]))
                continue
            s = score_peptide(
                model, seq, target, target_names, device, feature_dim,
            )
            if s is not None:
                all_seen[seq] = s
                scored.append((seq, s))
        return scored

    def tournament_select(scored: List[Tuple[str, float]]) -> str:
        contestants = random.sample(scored, min(tournament_size, len(scored)))
        return max(contestants, key=lambda x: x[1])[0]

    for gen in range(n_generations):
        scored = evaluate_population(population)
        if not scored:
            population = [random_sequence(length) for _ in range(population_size)]
            continue

        scored.sort(key=lambda x: x[1], reverse=True)

        # Elite carry-over
        n_elite = max(int(population_size * elite_frac), 1)
        next_pop = [s for s, _ in scored[:n_elite]]

        # Breed the rest
        while len(next_pop) < population_size:
            parent_a = tournament_select(scored)
            parent_b = tournament_select(scored)
            child_a, child_b = crossover(parent_a, parent_b)

            # Mutate
            if random.random() < mutation_rate:
                child_a = mutate_single(child_a)
            if random.random() < mutation_rate:
                child_b = mutate_single(child_b)

            next_pop.append(child_a)
            if len(next_pop) < population_size:
                next_pop.append(child_b)

        population = next_pop

        if (gen + 1) % max(n_generations // 5, 1) == 0:
            best_score = scored[0][1]
            mean_score = sum(s for _, s in scored) / len(scored)
            print(
                f"  GA gen {gen + 1}/{n_generations} | "
                f"best: {best_score:.3f} | mean: {mean_score:.3f} | "
                f"unique: {len(all_seen)}"
            )

    # Final ranking of everything we have ever seen
    ranked = sorted(all_seen.items(), key=lambda x: x[1], reverse=True)
    return ranked[:n_candidates]


# ------------------------------------------------------------------
# Strategy 3: Exhaustive Enumeration
# ------------------------------------------------------------------

def generate_exhaustive(
    model: MealShieldGNN,
    target: str,
    target_names: List[str],
    device: torch.device,
    feature_dim: int,
    length: int = 2,
    n_candidates: int = 100,
) -> List[Tuple[str, float]]:
    """Exhaustive enumeration of all 20^n peptides (practical for n <= 3).

    Scores every possible peptide of the given length and returns the top-K.
    """
    total = len(AMINO_ACIDS) ** length
    if length > 4:
        print(
            f"  WARNING: Exhaustive enumeration for length {length} would "
            f"require scoring {total:,} sequences. This may be very slow."
        )
        ans = input("  Continue? [y/N] ").strip().lower()
        if ans != "y":
            print("  Aborted.")
            return []

    print(f"  Enumerating all {total:,} {length}-mers...")

    scored: List[Tuple[str, float]] = []
    count = 0
    t0 = time.time()

    for combo in product(AMINO_ACIDS, repeat=length):
        seq = "".join(combo)
        s = score_peptide(
            model, seq, target, target_names, device, feature_dim,
        )
        if s is not None:
            scored.append((seq, s))

        count += 1
        if count % 1000 == 0:
            elapsed = time.time() - t0
            rate = count / max(elapsed, 0.01)
            remaining = (total - count) / max(rate, 0.01)
            print(
                f"  Scored {count:,}/{total:,} "
                f"({100 * count / total:.1f}%) | "
                f"~{remaining:.0f}s remaining"
            )

    elapsed = time.time() - t0
    print(f"  Enumeration complete: {count:,} scored in {elapsed:.1f}s")

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n_candidates]


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model(
    checkpoint_path: str = "checkpoints/meal_shield_gnn.pt",
) -> Tuple[MealShieldGNN, List[str], int, torch.device]:
    """Load trained model from checkpoint.

    Returns:
        model: The loaded MealShieldGNN in mode.
        target_names: List of target names from the checkpoint.
        feature_dim: Node feature dimensionality.
        device: The device the model is on.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Train the model first:  python train.py --esm --multitask")
        sys.exit(1)

    # MPS doesn't implement scatter_reduce for GAT — force CPU on macOS unless CUDA is present.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    feature_dim = checkpoint["feature_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    target_names = checkpoint["target_names"]

    model = MealShieldGNN(
        node_feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        target_names=target_names,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Device: {device}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Targets: {target_names}")

    return model, target_names, feature_dim, device


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

def pIC50_to_IC50_uM(pic50: float) -> float:
    """Convert pIC50 to IC50 in micromolar."""
    if pic50 > 0:
        ic50_nM = 10 ** (9 - pic50)
        return ic50_nM / 1000.0
    return float("inf")


def print_results(
    ranked: List[Tuple[str, float]],
    target: str,
    top_k: int,
) -> List[Dict]:
    """Pretty-print the top-K results and return them as a list of dicts."""
    display = ranked[:top_k]
    print(f"\n{'='*64}")
    print(f"Top {len(display)} peptides for target: {target}")
    print(f"{'='*64}")
    print(f"{'Rank':>5}  {'Sequence':<12} {'pIC50':>8} {'IC50_uM':>12}")
    print(f"{'-'*5}  {'-'*12} {'-'*8} {'-'*12}")

    rows = []
    for rank, (seq, pic50) in enumerate(display, 1):
        ic50 = pIC50_to_IC50_uM(pic50)
        ic50_str = f"{ic50:.2f}" if ic50 < 1e6 else "inf"
        print(f"{rank:>5}  {seq:<12} {pic50:>8.3f} {ic50_str:>12}")
        rows.append({
            "rank": rank,
            "sequence": seq,
            "target": target,
            "pIC50": round(pic50, 4),
            "IC50_uM": round(ic50, 4) if ic50 < 1e6 else None,
            "length": len(seq),
        })
    return rows


def save_results(rows: List[Dict], output_path: str) -> None:
    """Save generated peptide results to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["rank", "sequence", "target", "pIC50", "IC50_uM", "length"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {output_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate optimized peptide sequences using the trained GNN as a scoring oracle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python generate.py --target ace_inhibitor --method mc --length 5 --n-candidates 100 --top-k 10
  python generate.py --target dpp4_inhibitor --method ga --length 4 --top-k 20
  python generate.py --target alpha_glucosidase --method exhaustive --length 3 --top-k 15
""",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Bioactivity target to optimize for (e.g. ace_inhibitor, dpp4_inhibitor).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mc", "ga", "exhaustive"],
        default="mc",
        help="Generation strategy: mc (Monte Carlo), ga (genetic algorithm), exhaustive.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=5,
        help="Peptide length (number of residues). Default: 5.",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=100,
        help="Number of candidate sequences to generate/retain. Default: 100.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to display and save. Default: 10.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/meal_shield_gnn.pt",
        help="Path to model checkpoint. Default: checkpoints/meal_shield_gnn.pt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated_peptides.csv",
        help="Output CSV path. Default: data/generated_peptides.csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )

    # MC-specific
    parser.add_argument(
        "--mc-restarts",
        type=int,
        default=10,
        help="(MC) Number of independent restarts. Default: 10.",
    )
    parser.add_argument(
        "--mc-t-start",
        type=float,
        default=1.0,
        help="(MC) Initial annealing temperature. Default: 1.0.",
    )
    parser.add_argument(
        "--mc-t-end",
        type=float,
        default=0.01,
        help="(MC) Final annealing temperature. Default: 0.01.",
    )

    # GA-specific
    parser.add_argument(
        "--ga-pop-size",
        type=int,
        default=50,
        help="(GA) Population size. Default: 50.",
    )
    parser.add_argument(
        "--ga-generations",
        type=int,
        default=40,
        help="(GA) Number of generations. Default: 40.",
    )
    parser.add_argument(
        "--ga-mutation-rate",
        type=float,
        default=0.3,
        help="(GA) Mutation rate per offspring. Default: 0.3.",
    )
    parser.add_argument(
        "--ga-tournament-size",
        type=int,
        default=3,
        help="(GA) Tournament selection size. Default: 3.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    print("=" * 64)
    print("Meal Shield GNN -- Peptide Generator")
    print("=" * 64)

    model, target_names, feature_dim, device = load_model(args.checkpoint)

    # Validate target
    if args.target not in target_names:
        print(f"\nError: Unknown target '{args.target}'")
        print(f"Available targets: {target_names}")
        sys.exit(1)

    print(f"\nTarget: {args.target}")
    print(f"Method: {args.method}")
    print(f"Peptide length: {args.length}")
    print(f"Candidates to generate: {args.n_candidates}")
    print(f"Top-K to report: {args.top_k}")
    print()

    t0 = time.time()

    # Run selected strategy
    if args.method == "mc":
        print("Running Monte Carlo (simulated annealing)...")
        ranked = generate_monte_carlo(
            model=model,
            target=args.target,
            target_names=target_names,
            device=device,
            feature_dim=feature_dim,
            length=args.length,
            n_candidates=args.n_candidates,
            n_restarts=args.mc_restarts,
            t_start=args.mc_t_start,
            t_end=args.mc_t_end,
        )
    elif args.method == "ga":
        print("Running Genetic Algorithm...")
        ranked = generate_genetic(
            model=model,
            target=args.target,
            target_names=target_names,
            device=device,
            feature_dim=feature_dim,
            length=args.length,
            n_candidates=args.n_candidates,
            population_size=args.ga_pop_size,
            n_generations=args.ga_generations,
            mutation_rate=args.ga_mutation_rate,
            tournament_size=args.ga_tournament_size,
        )
    elif args.method == "exhaustive":
        print("Running Exhaustive Enumeration...")
        ranked = generate_exhaustive(
            model=model,
            target=args.target,
            target_names=target_names,
            device=device,
            feature_dim=feature_dim,
            length=args.length,
            n_candidates=args.n_candidates,
        )
    else:
        print(f"Error: Unknown method '{args.method}'")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\nGeneration complete in {elapsed:.1f}s ({len(ranked)} unique sequences)")

    if not ranked:
        print("No valid peptides were generated. Check the model and target.")
        sys.exit(1)

    # Display and save results
    rows = print_results(ranked, args.target, args.top_k)
    save_results(rows, args.output)

    # Summary statistics
    all_scores = [s for _, s in ranked]
    print(f"\nStatistics over all {len(ranked)} candidates:")
    print(f"  Best pIC50:  {max(all_scores):.4f}")
    print(f"  Worst pIC50: {min(all_scores):.4f}")
    print(f"  Mean pIC50:  {sum(all_scores) / len(all_scores):.4f}")
    best_seq, best_score = ranked[0]
    best_ic50 = pIC50_to_IC50_uM(best_score)
    print(f"\nBest candidate: {best_seq} (pIC50={best_score:.4f}, IC50={best_ic50:.2f} uM)")


# Aliases used by design.py and server.py
generate_mc = generate_monte_carlo
generate_enumerate = generate_exhaustive


if __name__ == "__main__":
    main()
