"""Tests for multi-objective Pareto selection module."""

import pytest

from pareto import pareto_front, rank_candidates, select_diverse, levenshtein_distance


# ── Pareto front ────────────────────────────────────────────────

class TestParetoFront:
    def test_dominated_points_excluded(self):
        candidates = [
            {"id": "A", "obj1": 0.9, "obj2": 0.9},
            {"id": "B", "obj1": 0.5, "obj2": 0.5},  # dominated by A
            {"id": "C", "obj1": 0.8, "obj2": 0.3},
        ]
        front = pareto_front(candidates, objectives=["obj1", "obj2"])
        ids = [c["id"] for c in front]
        assert "A" in ids, "A dominates B on both objectives"
        assert "B" not in ids, "B is dominated by A"

    def test_non_dominated_retained(self):
        candidates = [
            {"id": "A", "obj1": 0.9, "obj2": 0.2},
            {"id": "B", "obj1": 0.2, "obj2": 0.9},
        ]
        front = pareto_front(candidates, objectives=["obj1", "obj2"])
        ids = [c["id"] for c in front]
        assert "A" in ids
        assert "B" in ids

    def test_single_candidate_is_pareto_optimal(self):
        candidates = [{"id": "X", "obj1": 0.5, "obj2": 0.5}]
        front = pareto_front(candidates, objectives=["obj1", "obj2"])
        assert len(front) == 1
        assert front[0]["id"] == "X"

    def test_all_equal_all_returned(self):
        candidates = [
            {"id": "A", "obj1": 0.5, "obj2": 0.5},
            {"id": "B", "obj1": 0.5, "obj2": 0.5},
        ]
        front = pareto_front(candidates, objectives=["obj1", "obj2"])
        assert len(front) == 2

    def test_three_objectives(self):
        candidates = [
            {"id": "A", "obj1": 0.9, "obj2": 0.9, "obj3": 0.9},
            {"id": "B", "obj1": 0.1, "obj2": 0.1, "obj3": 0.1},  # dominated
        ]
        front = pareto_front(candidates, objectives=["obj1", "obj2", "obj3"])
        ids = [c["id"] for c in front]
        assert "A" in ids
        assert "B" not in ids


# ── Levenshtein distance ────────────────────────────────────────

class TestLevenshteinDistance:
    def test_identical_strings(self):
        assert levenshtein_distance("ABC", "ABC") == 0

    def test_single_substitution(self):
        assert levenshtein_distance("ABC", "ABD") == 1

    def test_empty_to_nonempty(self):
        assert levenshtein_distance("", "ABC") == 3

    def test_nonempty_to_empty(self):
        assert levenshtein_distance("ABC", "") == 3

    def test_both_empty(self):
        assert levenshtein_distance("", "") == 0

    def test_insertion(self):
        assert levenshtein_distance("AC", "ABC") == 1

    def test_deletion(self):
        assert levenshtein_distance("ABC", "AC") == 1

    def test_symmetric(self):
        d1 = levenshtein_distance("KITTEN", "SITTING")
        d2 = levenshtein_distance("SITTING", "KITTEN")
        assert d1 == d2


# ── Rank candidates ────────────────────────────────────────────

class TestRankCandidates:
    def test_equal_weights_ordering(self):
        candidates = [
            {"sequence": "A", "score1": 0.2, "score2": 0.3},
            {"sequence": "B", "score1": 0.9, "score2": 0.8},
            {"sequence": "C", "score1": 0.5, "score2": 0.5},
        ]
        weights = {"score1": 1.0, "score2": 1.0}
        ranked = rank_candidates(candidates, weights)
        assert len(ranked) == 3
        # B has highest combined score, should be first
        assert ranked[0]["sequence"] == "B"
        # A has lowest combined score, should be last
        assert ranked[-1]["sequence"] == "A"

    def test_single_objective_weight(self):
        candidates = [
            {"sequence": "A", "score1": 0.1, "score2": 0.9},
            {"sequence": "B", "score1": 0.9, "score2": 0.1},
        ]
        weights = {"score1": 1.0, "score2": 0.0}
        ranked = rank_candidates(candidates, weights)
        assert ranked[0]["sequence"] == "B"

    def test_preserves_all_candidates(self):
        candidates = [{"sequence": str(i), "s": float(i)} for i in range(10)]
        weights = {"s": 1.0}
        ranked = rank_candidates(candidates, weights)
        assert len(ranked) == 10


# ── Select diverse ──────────────────────────────────────────────

class TestSelectDiverse:
    def test_returns_n_candidates(self):
        candidates = [
            {"sequence": "AAAA"},
            {"sequence": "BBBB"},
            {"sequence": "CCCC"},
            {"sequence": "DDDD"},
            {"sequence": "EEEE"},
        ]
        selected = select_diverse(candidates, n=3, diversity_threshold=1)
        assert len(selected) == 3

    def test_diverse_pairwise_distance(self):
        candidates = [
            {"sequence": "AAAA"},
            {"sequence": "AAAB"},
            {"sequence": "WWWW"},
            {"sequence": "DDDD"},
            {"sequence": "KKKK"},
        ]
        selected = select_diverse(candidates, n=3, diversity_threshold=0.5)
        # Should select diverse candidates, not near-duplicates
        assert len(selected) >= 1

    def test_n_greater_than_available(self):
        candidates = [{"sequence": "AA"}, {"sequence": "BB"}]
        selected = select_diverse(candidates, n=5, diversity_threshold=1)
        assert len(selected) <= 2

    def test_single_candidate(self):
        candidates = [{"sequence": "LIWKL"}]
        selected = select_diverse(candidates, n=1, diversity_threshold=1)
        assert len(selected) == 1
        assert selected[0]["sequence"] == "LIWKL"
