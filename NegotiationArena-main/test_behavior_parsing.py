#!/usr/bin/env python3
"""
Test script to verify parsing of new behavior names (Marwadi_Forced and Baseline)
"""

import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from analyze_real_results import UltimatumResultsAnalyzer


def test_parsing():
    """Test directory name parsing for new behaviors"""

    analyzer = UltimatumResultsAnalyzer(".")

    # Test cases with new behaviors
    test_cases = [
        # Format: (directory_name, expected_model1, expected_model2, expected_behavior, expected_iteration)
        (
            "GPT-4o_vs_GPT-3.5_Marwadi_Forced_iter_1",
            "GPT-4o",
            "GPT-3.5",
            "Marwadi_Forced",
            1,
        ),
        (
            "GPT-3.5_vs_GPT-4o_Marwadi_Forced_iter_2",
            "GPT-3.5",
            "GPT-4o",
            "Marwadi_Forced",
            2,
        ),
        (
            "GPT-4o_vs_Claude-3.5-Haiku_Marwadi_Forced_iter_3",
            "GPT-4o",
            "Claude-3.5-Haiku",
            "Marwadi_Forced",
            3,
        ),
        (
            "Claude-3.5-Haiku_vs_GPT-oss_Marwadi_Forced_iter_4",
            "Claude-3.5-Haiku",
            "GPT-oss",
            "Marwadi_Forced",
            4,
        ),
        ("GPT-4o_vs_GPT-3.5_Baseline_iter_1", "GPT-4o", "GPT-3.5", "Baseline", 1),
        ("GPT-3.5_vs_GPT-4o_Baseline_iter_2", "GPT-3.5", "GPT-4o", "Baseline", 2),
        (
            "GPT-oss_vs_Claude-3.5-Haiku_Baseline_iter_3",
            "GPT-oss",
            "Claude-3.5-Haiku",
            "Baseline",
            3,
        ),
        (
            "Claude-3.5-Haiku_vs_GPT-4o_Baseline_iter_5",
            "Claude-3.5-Haiku",
            "GPT-4o",
            "Baseline",
            5,
        ),
        # Old behaviors should still work
        ("GPT-4o_vs_GPT-3.5_Hindi_iter_1", "GPT-4o", "GPT-3.5", "Hindi", 1),
        ("GPT-4o_vs_GPT-3.5_Gujarati_iter_1", "GPT-4o", "GPT-3.5", "Gujarati", 1),
        ("GPT-4o_vs_GPT-3.5_Marwadi_iter_1", "GPT-4o", "GPT-3.5", "Marwadi", 1),
        ("GPT-4o_vs_GPT-3.5_Punjabi_iter_1", "GPT-4o", "GPT-3.5", "Punjabi", 1),
    ]

    print("=" * 80)
    print("TESTING BEHAVIOR PARSING")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for (
        dir_name,
        expected_m1,
        expected_m2,
        expected_behavior,
        expected_iter,
    ) in test_cases:
        print(f"Testing: {dir_name}")
        model1, model2, behavior, iteration = analyzer.parse_directory_name(dir_name)

        success = (
            model1 == expected_m1
            and model2 == expected_m2
            and behavior == expected_behavior
            and iteration == expected_iter
        )

        if success:
            print(f"  ✓ PASSED")
            passed += 1
        else:
            print(f"  ✗ FAILED")
            print(
                f"    Expected: {expected_m1} vs {expected_m2} | {expected_behavior} | iter {expected_iter}"
            )
            print(f"    Got:      {model1} vs {model2} | {behavior} | iter {iteration}")
            failed += 1

        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print()

    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(test_parsing())
