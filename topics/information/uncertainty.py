"""Finite discrete entropy in bits, with category contributions.

Validation is adapted from the project's Shanny discrete_info helper.
Here accepted probabilities retain their supplied masses: no automatic
normalisation takes place. This computes one distribution's entropy,
not an entropy rate or a measure of meaning.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EntropyBreakdown:
    """Probabilities and their contributions share the supplied order."""

    probabilities: tuple[float, ...]
    contributions_bits: tuple[float, ...]
    entropy_bits: float


def _finite_number(
        value: float,
        label: str,
    ) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be an int or float, excluding bool.")

    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(
                f"{label} must be representable as a float."
            ) from error

    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")

    return number


def _probabilities(
        probabilities: Sequence[float],
        absolute_tolerance: float,
    ) -> tuple[float, ...]:
    tolerance = _finite_number(
            absolute_tolerance,
            "absolute_tolerance",
        )

    if not 0.0 <= tolerance < 1.0:
        raise ValueError("absolute_tolerance must lie in [0, 1).")

    values = tuple(
            _finite_number(value, f"probability[{index}]")
            for index, value in enumerate(probabilities)
        )

    if not values:
        raise ValueError("A probability distribution must not be empty.")

    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Each probability must lie in [0, 1].")

    total = math.fsum(values)

    if abs(total - 1.0) > tolerance:
        raise ValueError(
                f"Probability total {total!r} is outside the absolute "
                f"tolerance {tolerance!r} of one."
            )

    return values


def entropy_breakdown(
        probabilities: Sequence[float],
        *,
        absolute_tolerance: float = 1e-12,
    ) -> EntropyBreakdown:
    """Return -sum(p log2 p), with one contribution per category.

    Require a nonempty distribution of finite int/float probabilities in
    [0, 1], excluding bool. Its total must be within absolute_tolerance
    of one; require a finite tolerance in [0, 1). Invalid types raise
    TypeError and invalid values raise ValueError.

    Exactly zero contributes zero by continuous extension. Positive masses
    are never clipped or renormalised, and inputs are not mutated. Small
    accepted total errors therefore remain in the returned calculation.
    Counts must be converted to probabilities explicitly by the caller.
    """
    values = _probabilities(
            probabilities,
            absolute_tolerance,
        )
    contributions = tuple(
            0.0 if probability == 0.0
            else -probability * math.log2(probability)
            for probability in values
        )

    return EntropyBreakdown(
            values,
            contributions,
            math.fsum(contributions),
        )
