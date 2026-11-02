"""Numerical checks and domain failures for entropy in bits."""

import math
import unittest

from uncertainty import entropy_breakdown


class EntropyTests(unittest.TestCase):
    def test_uniform_four_way_distribution_has_two_bits(self) -> None:
        result = entropy_breakdown([0.25] * 4)

        self.assertEqual(result.entropy_bits, 2.0)
        self.assertEqual(result.contributions_bits, (0.5,) * 4)

    def test_known_distribution_exposes_each_contribution(self) -> None:
        result = entropy_breakdown([0.125, 0.125, 0.25, 0.5])

        self.assertEqual(result.entropy_bits, 1.75)
        self.assertEqual(
                result.contributions_bits,
                (0.375, 0.375, 0.5, 0.5),
            )

    def test_zero_extension_preserves_category_positions(self) -> None:
        result = entropy_breakdown([0.0, 1.0, 0.0])

        self.assertEqual(result.entropy_bits, 0.0)
        self.assertEqual(result.probabilities, (0.0, 1.0, 0.0))
        self.assertEqual(result.contributions_bits, (0.0, 0.0, 0.0))

    def test_small_positive_mass_is_not_clipped(self) -> None:
        smallest_positive = math.ulp(0.0)
        result = entropy_breakdown([smallest_positive, 1.0])

        self.assertGreater(result.contributions_bits[0], 0.0)
        self.assertEqual(result.probabilities[0], smallest_positive)

    def test_bits_agree_with_independent_natural_logs(self) -> None:
        probabilities = [0.25, 0.75]
        entropy_nats = -sum(
                probability * math.log(probability)
                for probability in probabilities
            )
        result = entropy_breakdown(probabilities)

        self.assertAlmostEqual(
                result.entropy_bits * math.log(2.0),
                entropy_nats,
                places=14,
            )

    def test_accepted_masses_are_neither_normalised_nor_mutated(self) -> None:
        supplied = [0.25, 0.7500000000001]
        original = supplied.copy()
        result = entropy_breakdown(supplied)

        self.assertEqual(supplied, original)
        self.assertEqual(result.probabilities, tuple(original))
        self.assertEqual(
                result.contributions_bits[0],
                -original[0] * math.log2(original[0]),
            )
        self.assertNotEqual(
                result.probabilities[0],
                supplied[0] / math.fsum(supplied),
            )

    def test_zero_tolerance_rejects_total_rounding_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Probability total"):
            entropy_breakdown(
                    [0.25, 0.7500000000001],
                    absolute_tolerance=0.0,
                )

    def test_invalid_distributions_are_rejected(self) -> None:
        for values in [
                [],
                [0.0, 0.0],
                [0.25, 0.25],
                [-0.25, 1.0],
                [1.25],
                [math.inf],
                [math.nan],
                [10 ** 1_000],
            ]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    entropy_breakdown(values)

    def test_invalid_probability_types_are_rejected(self) -> None:
        for values in [[True], ["1"], [None]]:
            with self.subTest(values=values):
                with self.assertRaises(TypeError):
                    entropy_breakdown(values)

    def test_invalid_tolerances_are_rejected(self) -> None:
        for tolerance in [-1e-12, 1.0, math.inf, math.nan]:
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    entropy_breakdown(
                            [1.0],
                            absolute_tolerance=tolerance,
                        )

        with self.assertRaises(TypeError):
            entropy_breakdown([1.0], absolute_tolerance=True)


if __name__ == "__main__":
    unittest.main()
