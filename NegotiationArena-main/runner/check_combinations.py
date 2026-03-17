#!/usr/bin/env python3
"""
Check that all expected model x language x iteration combinations
are present as subdirectories in a given logs folder.

Usage:
    python check_combinations.py <logs_path>

Expected folder format:
    {model1}_vs_{model2}_{language}_iter_{n}

e.g.  Llama-3.3-70B_vs_GPT-4o_Hindi_iter_6
"""

import os
import sys
from itertools import permutations

# ── Configure models and languages here ──────────────────────────────────────

MODELS = [
    "GPT-4o",
    "Claude-3-Haiku",
    "Claude-3.5-Haiku",
    "Llama-3.3-70B",
    "GPT-3.5",
]

LANGUAGES = [
    "Hindi",
    "Gujarati",
    "Punjabi",
    "Marwadi",
    "Baseline",
]

ITERATIONS = list(range(1, 31))  # 1 to 30 inclusive

# ─────────────────────────────────────────────────────────────────────────────


def expected_name(model1: str, model2: str, language: str, iteration: int) -> str:
    return f"{model1}_vs_{model2}_{language}_iter_{iteration}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_combinations.py <logs_path>")
        sys.exit(1)

    logs_path = sys.argv[1]

    if not os.path.isdir(logs_path):
        print(f"ERROR: '{logs_path}' is not a directory or does not exist.")
        sys.exit(1)

    # Collect all subdirectory names
    existing = {
        entry.name
        for entry in os.scandir(logs_path)
        if entry.is_dir()
    }

    print(f"Logs path   : {logs_path}")
    print(f"Models      : {MODELS}")
    print(f"Languages   : {LANGUAGES}")
    print(f"Iterations  : {ITERATIONS[0]} – {ITERATIONS[-1]}")
    print(f"Existing dirs: {len(existing)}")
    print()

    # All ordered pairs excluding same-model
    model_pairs = list(permutations(MODELS, 2))
    total_expected = len(model_pairs) * len(LANGUAGES) * len(ITERATIONS)

    missing = []
    present = 0

    for model1, model2 in model_pairs:
        for language in LANGUAGES:
            for iteration in ITERATIONS:
                name = expected_name(model1, model2, language, iteration)
                if name in existing:
                    present += 1
                else:
                    missing.append((model1, model2, language, iteration))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"Expected : {total_expected}")
    print(f"Present  : {present}")
    print(f"Missing  : {len(missing)}")
    print()

    if not missing:
        print("✓ All combinations are present!")
        return

    print("✗ Missing combinations:")
    print("-" * 60)

    # Group by (model1, model2, language) for readability
    grouped: dict[tuple, list[int]] = {}
    for model1, model2, language, iteration in missing:
        key = (model1, model2, language)
        grouped.setdefault(key, []).append(iteration)

    for (model1, model2, language), iters in sorted(grouped.items()):
        print(f"  {model1}_vs_{model2}_{language}")
        print(f"    Missing iters: {sorted(iters)}")

    print()
    print(f"Total missing: {len(missing)} / {total_expected}")


if __name__ == "__main__":
    main()
