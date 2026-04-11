"""
Unit tests for ClusteringFeature and cosine_similarity.

All functions are pure (no I/O, no DB) — no mocks needed.
"""
import numpy as np
import pytest

from services.clustering.birch import ClusteringFeature, cosine_similarity


# ── Helpers ───────────────────────────────────────────────────────────────────

def unit_vec(dim: int = 4, idx: int = 0) -> np.ndarray:
    """Return a unit vector with a 1.0 at position idx, zeros elsewhere."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


# ── ClusteringFeature.from_point ──────────────────────────────────────────────

class TestFromPoint:
    def test_n_is_one(self):
        cf = ClusteringFeature.from_point(unit_vec())
        assert cf.n == 1

    def test_ls_equals_point(self):
        p = unit_vec(idx=2)
        cf = ClusteringFeature.from_point(p)
        np.testing.assert_array_equal(cf.ls, p)

    def test_ss_is_squared_norm(self):
        p = np.array([3.0, 4.0], dtype=np.float32)
        cf = ClusteringFeature.from_point(p)
        assert cf.ss == pytest.approx(25.0)  # 3² + 4²

    def test_centroid_equals_point_for_single_point(self):
        p = unit_vec()
        cf = ClusteringFeature.from_point(p)
        np.testing.assert_array_almost_equal(cf.centroid, p)

    def test_radius_is_zero_for_single_point(self):
        cf = ClusteringFeature.from_point(unit_vec())
        assert cf.radius == pytest.approx(0.0)


# ── ClusteringFeature.absorb ──────────────────────────────────────────────────

class TestAbsorb:
    def test_n_increments(self):
        cf = ClusteringFeature.from_point(unit_vec(idx=0)).absorb(unit_vec(idx=1))
        assert cf.n == 2

    def test_ls_is_sum_of_points(self):
        p1, p2 = unit_vec(idx=0), unit_vec(idx=1)
        cf = ClusteringFeature.from_point(p1).absorb(p2)
        np.testing.assert_array_almost_equal(cf.ls, p1 + p2)

    def test_ss_accumulates(self):
        p1 = np.array([1.0, 0.0], dtype=np.float32)
        p2 = np.array([0.0, 2.0], dtype=np.float32)
        cf = ClusteringFeature.from_point(p1).absorb(p2)
        assert cf.ss == pytest.approx(1.0 + 4.0)

    def test_original_cf_is_unchanged(self):
        p1 = unit_vec(idx=0)
        cf = ClusteringFeature.from_point(p1)
        cf.absorb(unit_vec(idx=1))
        assert cf.n == 1  # absorb returns a new CF, original must not mutate

    def test_centroid_is_mean_of_points(self):
        p1 = np.array([1.0, 0.0], dtype=np.float32)
        p2 = np.array([0.0, 1.0], dtype=np.float32)
        cf = ClusteringFeature.from_point(p1).absorb(p2)
        np.testing.assert_array_almost_equal(cf.centroid, [0.5, 0.5])

    def test_multiple_absorbs_chain_correctly(self):
        cf = ClusteringFeature.from_point(unit_vec(dim=3, idx=0))
        cf = cf.absorb(unit_vec(dim=3, idx=1))
        cf = cf.absorb(unit_vec(dim=3, idx=2))
        assert cf.n == 3
        np.testing.assert_array_almost_equal(cf.centroid, [1/3, 1/3, 1/3])


# ── ClusteringFeature.would_absorb ───────────────────────────────────────────

class TestWouldAbsorb:
    def test_identical_point_absorbed_at_tight_threshold(self):
        p = unit_vec()
        cf = ClusteringFeature.from_point(p)
        assert cf.would_absorb(p, threshold=0.01)

    def test_orthogonal_point_rejected_at_tight_threshold(self):
        p1, p2 = unit_vec(idx=0), unit_vec(idx=1)
        cf = ClusteringFeature.from_point(p1)
        assert not cf.would_absorb(p2, threshold=0.05)

    def test_orthogonal_point_accepted_at_loose_threshold(self):
        p1, p2 = unit_vec(idx=0), unit_vec(idx=1)
        cf = ClusteringFeature.from_point(p1)
        assert cf.would_absorb(p2, threshold=1.0)

    def test_consistent_with_radius_after_absorb(self):
        """would_absorb(p, t) must agree with the actual radius after absorbing p."""
        p1 = unit_vec(dim=8, idx=0)
        p2 = unit_vec(dim=8, idx=1)
        threshold = 0.5
        cf = ClusteringFeature.from_point(p1)
        decision = cf.would_absorb(p2, threshold)
        actual_radius = cf.absorb(p2).radius
        assert (actual_radius <= threshold) == decision

    def test_threshold_boundary(self):
        """A point that brings radius exactly to threshold must be accepted."""
        p1 = np.array([1.0, 0.0], dtype=np.float32)
        p2 = np.array([0.0, 1.0], dtype=np.float32)
        cf = ClusteringFeature.from_point(p1)
        radius_after = cf.absorb(p2).radius
        assert cf.would_absorb(p2, threshold=radius_after)
        assert not cf.would_absorb(p2, threshold=radius_after - 1e-5)


# ── cosine_similarity ─────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        zero = np.array([0.0, 0.0], dtype=np.float32)
        v = np.array([1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(zero, v) == pytest.approx(0.0)

    def test_scale_invariant(self):
        a = np.array([1.0, 1.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a * 5, b * 100) == pytest.approx(1.0)

    def test_known_angle(self):
        # 45-degree angle between [1,0] and [1,1]/sqrt(2) → cos(45°) ≈ 0.707
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(1 / np.sqrt(2), abs=1e-5)
