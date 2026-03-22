import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from postprocess import (  # noqa: E402
    _convert_unit_for_question,
    _detect_output_format,
    _extract_percent_answer,
    _is_placeholder_answer,
    _looks_like_list_answer,
)


class PostprocessUnitTests(unittest.TestCase):
    def test_officeqa_format_detection_ignores_options_keyword(self) -> None:
        task = (
            "What was the difference between Japanese yen and British pound "
            "net options positions in March 2025?"
        )
        fmt_key, directive = _detect_output_format(task, task_type="officeqa")
        self.assertIsNone(fmt_key)
        self.assertEqual(directive, "")

    def test_placeholder_detection_catches_common_bad_answers(self) -> None:
        self.assertTrue(_is_placeholder_answer("nan"))
        self.assertTrue(_is_placeholder_answer("Not found in the provided reference data"))
        self.assertFalse(_is_placeholder_answer("2602"))

    def test_list_answer_detection_preserves_bracketed_numeric_pairs(self) -> None:
        self.assertTrue(_looks_like_list_answer("[10102000000, 4.73]"))
        self.assertTrue(_looks_like_list_answer("10102000000, 4.73"))
        self.assertFalse(_looks_like_list_answer("2602"))

    def test_percent_extraction_skips_years_and_uses_real_ratio(self) -> None:
        question = "Report the percent change as a percentage value."
        text = (
            "Base year 1940.\n"
            "The percent change between the two totals is 0.14.\n"
            "Do not return the year."
        )
        self.assertEqual(_extract_percent_answer(question, text), "14%")

    def test_unit_conversion_requires_explicit_source_unit(self) -> None:
        question = "Report your answer in billions."
        self.assertEqual(_convert_unit_for_question(question, "129124"), "129124")
        self.assertEqual(_convert_unit_for_question(question, "129124 million"), "129.124")


if __name__ == "__main__":
    unittest.main()
