"""Distances and exhaustive bounded-error checks for small codes."""

from itertools import product
import unittest

from block_codes import (
        code_guarantees,
        hamming_distance,
    )


class HammingDistanceTests(unittest.TestCase):
    def test_a_word_has_zero_distance_from_itself(self) -> None:
        self.assertEqual(hamming_distance("10101", "10101"), 0)

    def test_known_distance_is_symmetric(self) -> None:
        self.assertEqual(hamming_distance("0010", "1111"), 3)
        self.assertEqual(hamming_distance("1111", "0010"), 3)

    def test_different_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            hamming_distance("0", "01")

    def test_empty_and_nonbinary_words_are_rejected(self) -> None:
        for word in ["", "102", " 01", "01\n"]:
            with self.subTest(word=word):
                with self.assertRaises(ValueError):
                    hamming_distance(word, word)

        with self.assertRaises(TypeError):
            hamming_distance(10, "10")


class CodeGuaranteeTests(unittest.TestCase):
    def test_fresh_code_has_minimum_distance_three(self) -> None:
        supplied = ["00000", "11100", "10011"]
        result = code_guarantees(supplied)

        self.assertEqual(supplied, ["00000", "11100", "10011"])
        self.assertEqual(result.word_length, 5)
        self.assertEqual(result.codeword_count, 3)
        self.assertEqual(result.minimum_distance, 3)
        self.assertEqual(result.detectable_errors, 2)
        self.assertEqual(result.correctable_errors, 1)

    def test_adjacent_codewords_have_no_positive_error_guarantee(self) -> None:
        result = code_guarantees(["00", "01"])

        self.assertEqual(result.minimum_distance, 1)
        self.assertEqual(result.detectable_errors, 0)
        self.assertEqual(result.correctable_errors, 0)

    def test_three_bit_repetition_code_guarantees(self) -> None:
        result = code_guarantees(["000", "111"])

        self.assertEqual(result.minimum_distance, 3)
        self.assertEqual(result.detectable_errors, 2)
        self.assertEqual(result.correctable_errors, 1)

    def test_repetition_code_detects_all_one_and_two_bit_errors(self) -> None:
        code = ("000", "111")
        guarantees = code_guarantees(code)

        for transmitted in code:
            for bits in product("01", repeat=3):
                received = "".join(bits)
                errors = hamming_distance(transmitted, received)

                if 0 < errors <= guarantees.detectable_errors:
                    with self.subTest(
                            transmitted=transmitted,
                            received=received,
                        ):
                        self.assertNotIn(received, code)

    def test_repetition_code_corrects_up_to_one_bit_error(self) -> None:
        code = ("000", "111")
        guarantees = code_guarantees(code)

        for transmitted in code:
            for bits in product("01", repeat=3):
                received = "".join(bits)
                errors = hamming_distance(transmitted, received)

                if errors > guarantees.correctable_errors:
                    continue

                distances = {
                        word: hamming_distance(word, received)
                        for word in code
                    }
                nearest_distance = min(distances.values())
                nearest_words = [
                        word
                        for word, distance in distances.items()
                        if distance == nearest_distance
                    ]

                self.assertEqual(nearest_words, [transmitted])

    def test_three_errors_can_change_one_codeword_into_another(self) -> None:
        code = ("000", "111")
        received = "111"

        self.assertEqual(hamming_distance("000", received), 3)
        self.assertIn(received, code)

    def test_empty_singleton_and_duplicate_codes_are_rejected(self) -> None:
        for code in [[], ["000"], ["000", "000"]]:
            with self.subTest(code=code):
                with self.assertRaises(ValueError):
                    code_guarantees(code)

    def test_invalid_codewords_and_lengths_are_rejected(self) -> None:
        for code in [["0", "11"], ["", "1"], ["00", "02"]]:
            with self.subTest(code=code):
                with self.assertRaises(ValueError):
                    code_guarantees(code)

        for code in ["01", ["00", 11]]:
            with self.subTest(code=code):
                with self.assertRaises(TypeError):
                    code_guarantees(code)


if __name__ == "__main__":
    unittest.main()
