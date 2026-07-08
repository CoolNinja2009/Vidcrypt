#!/usr/bin/env python3
"""
Vidcrypt-V8 Test Runner

Runs all pure-Python tests with a common entry point.
Usage:  python run_tests.py
"""

import importlib
import sys
import time

TEST_MODULES = [
    "test_crc16",
    "test_bitstream",
    "test_reedsolomon",
    "test_decode",
]


def main() -> int:
    print("=" * 56)
    print("  VIDCRYPT-V8  --  Pure Python Test Suite")
    print("=" * 56)
    print()

    failures = 0
    total_start = time.perf_counter()

    for mod_name in TEST_MODULES:
        print(f"[{'=' * 48}]")
        print(f"  Module: {mod_name}")
        print(f"[{'=' * 48}]")
        print()

        try:
            mod = importlib.import_module(mod_name)
            start = time.perf_counter()
            result = mod.main()
            elapsed = time.perf_counter() - start
            status = "PASS" if result == 0 else "FAIL"
            print(f"\n  [{status}] {mod_name}  ({elapsed:.3f}s)")
            if result != 0:
                failures += 1
        except Exception as e:
            print(f"\n  [ERROR] {mod_name}: {e}")
            failures += 1

        print()

    total_elapsed = time.perf_counter() - total_start
    tested = len(TEST_MODULES) - failures

    print(f"[{'=' * 48}]")
    print(f"  {tested}/{len(TEST_MODULES)} tests passed  ({total_elapsed:.3f}s)")
    print(f"[{'=' * 48}]")

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
