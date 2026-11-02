"""Hamming distance and bounded-error guarantees for binary block codes.

Words are nonempty binary strings. Guarantees concern up to the returned
number of bit substitutions in one word, not at least that many errors.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class CodeGuarantees:
    word_length: int
    codeword_count: int
    minimum_distance: int
    detectable_errors: int
    correctable_errors: int


def _validate_word(
        word: str,
    ) -> None:
    if not isinstance(word, str):
        raise TypeError("A binary word must be a string.")

    if not word:
        raise ValueError("A binary word must not be empty.")

    if any(symbol not in "01" for symbol in word):
        raise ValueError("A binary word may contain only '0' and '1'.")


def _distance(
        first: str,
        second: str,
    ) -> int:
    return sum(
            first_bit != second_bit
            for first_bit, second_bit in zip(first, second)
        )


def hamming_distance(
        first: str,
        second: str,
    ) -> int:
    """Count differing positions in two nonempty equal-length words.

    Require binary strings; raise TypeError for other types and ValueError
    for empty, nonbinary or different-length words. The result lies between
    zero and the word length, with zero precisely for identical words.
    """
    _validate_word(first)
    _validate_word(second)

    if len(first) != len(second):
        raise ValueError("Binary words must have the same length.")

    return _distance(
            first,
            second,
        )


def _codewords(
        codewords: Iterable[str],
    ) -> tuple[str, ...]:
    if isinstance(codewords, str):
        raise TypeError("Supply a collection of words, not one string.")

    words = tuple(codewords)

    if len(words) < 2:
        raise ValueError("A code must contain at least two distinct words.")

    for word in words:
        _validate_word(word)

    if len(set(words)) != len(words):
        raise ValueError("Codewords must be distinct; remove duplicates.")

    length = len(words[0])

    if any(len(word) != length for word in words):
        raise ValueError("All codewords must have the same length.")

    return words


def code_guarantees(
        codewords: Iterable[str],
    ) -> CodeGuarantees:
    """Return minimum distance and guaranteed detection/correction radii.

    Require a finite collection of at least two distinct, nonempty binary
    strings of equal length. Invalid types raise TypeError; invalid values
    raise ValueError. Inputs are not mutated and pairs are checked exactly.

    With minimum distance d, every nonzero error of weight at most d - 1
    is detected. Every error of weight at most (d - 1) // 2 has a unique
    nearest codeword equal to the transmitted word. These are worst-case
    guarantees for substitutions, not probabilities or insertion/deletion
    guarantees. No decoder is run here.
    """
    words = _codewords(codewords)
    minimum_distance = min(
            _distance(first, second)
            for first, second in combinations(words, 2)
        )

    return CodeGuarantees(
            len(words[0]),
            len(words),
            minimum_distance,
            minimum_distance - 1,
            (minimum_distance - 1) // 2,
        )
