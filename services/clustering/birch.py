"""
BIRCH Clustering Feature (CF) domain logic.

CF = (N, LS, SS) where:
  N  = number of points in the cluster
  LS = linear sum of all embedding vectors
  SS = sum of squared norms (equals N for unit-norm vectors)

Derived quantities:
  centroid = LS / N
  radius   = sqrt(max(0, SS/N - ||LS/N||²))

All functions here are pure — no DB, no I/O. Testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusteringFeature:
    n: int
    ls: np.ndarray  # shape (D,), float32
    ss: float

    @property
    def centroid(self) -> np.ndarray:
        return self.ls / self.n

    @property
    def radius(self) -> float:
        ls_norm_sq = float(np.dot(self.ls, self.ls))
        r_sq = max(0.0, self.ss / self.n - ls_norm_sq / (self.n * self.n))
        return float(np.sqrt(r_sq))

    def would_absorb(self, point: np.ndarray, threshold: float) -> bool:
        """True if absorbing point keeps cluster radius <= threshold."""
        new_ls = self.ls + point
        new_n = self.n + 1
        new_ss = self.ss + float(np.dot(point, point))
        new_r_sq = max(
            0.0,
            new_ss / new_n - float(np.dot(new_ls, new_ls)) / (new_n * new_n),
        )
        return float(np.sqrt(new_r_sq)) <= threshold

    def absorb(self, point: np.ndarray) -> ClusteringFeature:
        """Return a new CF with point added. Original is unchanged."""
        return ClusteringFeature(
            n=self.n + 1,
            ls=self.ls + point,
            ss=self.ss + float(np.dot(point, point)),
        )

    @classmethod
    def from_point(cls, point: np.ndarray) -> ClusteringFeature:
        """Initialise a single-point cluster."""
        return cls(
            n=1,
            ls=point.copy(),
            ss=float(np.dot(point, point)),
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
