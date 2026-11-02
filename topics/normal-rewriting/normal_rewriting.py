"""Trace ordered Markov normal rewriting over Python strings.

The first applicable rule has priority. Within that rule we replace only
the leftmost occurrence. An empty left side matches position zero, so its
replacement is inserted before the current word. Positions count Unicode
code points; the caller defines the linguistic units and alphabet.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Rule:
    """A substitution, optionally terminal after its one application."""

    left: str
    right: str
    terminal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.left, str):
            raise TypeError("A rule's left side must be a string.")

        if not isinstance(self.right, str):
            raise TypeError("A rule's right side must be a string.")

        if not isinstance(self.terminal, bool):
            raise TypeError("A rule's terminal flag must be a bool.")


class HaltReason(Enum):
    NO_RULE = "no_rule"
    TERMINAL_RULE = "terminal_rule"
    STEP_LIMIT = "step_limit"


@dataclass(frozen=True)
class RewriteStep:
    """One substitution; rule_index and position are zero-based."""

    rule_index: int
    position: int
    before: str
    after: str


@dataclass(frozen=True)
class RewriteResult:
    word: str
    steps: tuple[RewriteStep, ...]
    halt_reason: HaltReason


def _first_match(
        word: str,
        rules: tuple[Rule, ...],
    ) -> tuple[int, int] | None:
    for index, rule in enumerate(rules):
        position = word.find(rule.left)

        if position >= 0:
            return index, position

    return None


def run_normal(
        word: str,
        rules: Iterable[Rule],
        *,
        max_steps: int = 1_000,
    ) -> RewriteResult:
    """Apply a finite ordered rule list and return its inspectable trace.

    Require a string, Rule objects and a nonnegative integer step budget
    (bool is excluded). Invalid domains raise TypeError or ValueError.
    Inputs are not mutated. Each trace entry performs exactly one rewrite.

    NO_RULE and TERMINAL_RULE are normal halts. STEP_LIMIT means another
    rule applies but its execution exceeds the budget; it does not prove
    divergence. A final normal halt is recognised even at exactly the
    budget, including zero steps when no rule applies.
    """
    if not isinstance(word, str):
        raise TypeError("The initial word must be a string.")

    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an integer, excluding bool.")

    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative.")

    ordered_rules = tuple(rules)

    if any(not isinstance(rule, Rule) for rule in ordered_rules):
        raise TypeError("Every rule must be a Rule object.")

    trace: list[RewriteStep] = []

    while True:
        match = _first_match(
                word,
                ordered_rules,
            )

        if match is None:
            return RewriteResult(
                    word,
                    tuple(trace),
                    HaltReason.NO_RULE,
                )

        if len(trace) == max_steps:
            return RewriteResult(
                    word,
                    tuple(trace),
                    HaltReason.STEP_LIMIT,
                )

        rule_index, position = match
        rule = ordered_rules[rule_index]
        rewritten = (
                word[:position]
                + rule.right
                + word[position + len(rule.left):]
            )
        trace.append(
                RewriteStep(
                        rule_index,
                        position,
                        word,
                        rewritten,
                    )
            )
        word = rewritten

        if rule.terminal:
            return RewriteResult(
                    word,
                    tuple(trace),
                    HaltReason.TERMINAL_RULE,
                )
