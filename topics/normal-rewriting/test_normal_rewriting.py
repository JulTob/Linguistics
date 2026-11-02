"""Operational cases for ordered rewriting, normal halts and bounds."""

import unittest

from normal_rewriting import (
        HaltReason,
        Rule,
        run_normal,
    )


class NormalRewritingTests(unittest.TestCase):
    def test_rule_order_precedes_position_in_the_word(self) -> None:
        result = run_normal(
                "ab",
                [
                    Rule("b", "B", terminal=True),
                    Rule("a", "A", terminal=True),
                    ],
            )

        self.assertEqual(result.word, "aB")
        self.assertEqual(result.steps[0].rule_index, 0)
        self.assertEqual(result.steps[0].position, 1)

    def test_first_rule_uses_only_its_leftmost_occurrence(self) -> None:
        result = run_normal(
                "ababa",
                [Rule("ba", "X", terminal=True)],
            )

        self.assertEqual(result.word, "aXba")
        self.assertEqual(result.steps[0].position, 1)

    def test_priority_restarts_after_each_substitution(self) -> None:
        result = run_normal(
                "c",
                [
                    Rule("b", "a"),
                    Rule("c", "b"),
                    ],
            )

        self.assertEqual(result.word, "a")
        self.assertEqual(result.halt_reason, HaltReason.NO_RULE)
        self.assertEqual(
                tuple(step.rule_index for step in result.steps),
                (1, 0),
            )
        self.assertEqual(
                tuple((step.before, step.after) for step in result.steps),
                (("c", "b"), ("b", "a")),
            )

    def test_terminal_rule_stops_while_another_rule_applies(self) -> None:
        result = run_normal(
                "a",
                [
                    Rule("a", "b", terminal=True),
                    Rule("b", "c"),
                    ],
            )

        self.assertEqual(result.word, "b")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.halt_reason, HaltReason.TERMINAL_RULE)

    def test_no_applicable_rule_is_a_zero_step_normal_halt(self) -> None:
        result = run_normal(
                "a",
                [Rule("b", "c")],
                max_steps=0,
            )

        self.assertEqual(result.word, "a")
        self.assertEqual(result.steps, ())
        self.assertEqual(result.halt_reason, HaltReason.NO_RULE)

    def test_empty_rule_list_is_a_normal_halt(self) -> None:
        result = run_normal("", [])

        self.assertEqual(result.word, "")
        self.assertEqual(result.halt_reason, HaltReason.NO_RULE)

    def test_zero_budget_does_not_execute_an_applicable_rule(self) -> None:
        result = run_normal(
                "a",
                [Rule("a", "b", terminal=True)],
                max_steps=0,
            )

        self.assertEqual(result.word, "a")
        self.assertEqual(result.steps, ())
        self.assertEqual(result.halt_reason, HaltReason.STEP_LIMIT)

    def test_identity_rule_consumes_budget_without_a_normal_halt(self) -> None:
        result = run_normal(
                "a",
                [Rule("a", "a")],
                max_steps=3,
            )

        self.assertEqual(result.word, "a")
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(result.halt_reason, HaltReason.STEP_LIMIT)

    def test_normal_halt_is_recognised_at_the_budget(self) -> None:
        for terminal, reason in [
                (False, HaltReason.NO_RULE),
                (True, HaltReason.TERMINAL_RULE),
            ]:
            with self.subTest(terminal=terminal):
                result = run_normal(
                        "a",
                        [Rule("a", "b", terminal=terminal)],
                        max_steps=1,
                    )

                self.assertEqual(result.word, "b")
                self.assertEqual(result.halt_reason, reason)

    def test_empty_left_side_inserts_at_position_zero(self) -> None:
        for word in ["", "ab"]:
            with self.subTest(word=word):
                result = run_normal(
                        word,
                        [Rule("", "x", terminal=True)],
                    )

                self.assertEqual(result.word, "x" + word)
                self.assertEqual(result.steps[0].position, 0)

    def test_nonterminal_empty_left_side_continues_to_the_bound(self) -> None:
        result = run_normal(
                "a",
                [Rule("", "x")],
                max_steps=2,
            )

        self.assertEqual(result.word, "xxa")
        self.assertEqual(result.halt_reason, HaltReason.STEP_LIMIT)

    def test_empty_right_side_deletes_the_matched_occurrence(self) -> None:
        result = run_normal(
                "aba",
                [Rule("a", "")],
            )

        self.assertEqual(result.word, "b")
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.halt_reason, HaltReason.NO_RULE)

    def test_invalid_rule_fields_and_rule_objects_are_rejected(self) -> None:
        for arguments in [(1, "a"), ("a", None), ("a", "b", 1)]:
            with self.subTest(arguments=arguments):
                with self.assertRaises(TypeError):
                    Rule(*arguments)

        with self.assertRaises(TypeError):
            run_normal("a", [("a", "b")])

    def test_invalid_initial_word_and_budgets_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            run_normal(1, [])

        for budget in [True, 1.5, "1"]:
            with self.subTest(budget=budget):
                with self.assertRaises(TypeError):
                    run_normal("a", [], max_steps=budget)

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            run_normal("a", [], max_steps=-1)


if __name__ == "__main__":
    unittest.main()
