"""Small statistics helpers for the evaluation report."""

from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar (sign test) p-value for discordant pair counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def pct_int(k: int, n: int) -> int:
    """k/n as a whole percentage, halves rounded up (62.5 -> 63), exactly, as the UI's Math.round does."""
    return (200 * k + n) // (2 * n) if n else 0


def round_half_up(x: float) -> int:
    """Round to the nearest integer with halves going up, like JavaScript's Math.round for positive numbers."""
    return math.floor(x + 0.5)
