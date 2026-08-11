#!/usr/bin/env python3
"""
Verbose test runner for the retrieval rework.

    python3 run_tests.py                # offline suite, grouped per-phase report
    python3 run_tests.py --phase 3      # just Phase 3 (BM25 + budget)
    python3 run_tests.py --llm          # add the live-Ollama tier
    python3 run_tests.py -k deferred    # pass anything else straight to pytest

The per-phase pass/fail table is printed by tests/conftest.py at the end of
the run, after pytest's own verbose per-test output.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PHASE_FILES: dict[str, list[str]] = {
    "0": ["test_config_context.py", "test_tokens.py", "test_runcontext.py", "test_errors.py"],
    "1": ["test_summaries.py", "test_index_migration.py"],
    "2": ["test_batched_prune.py"],
    "3": ["test_ranking.py", "test_budget_eval.py"],
    "4": ["test_manifest_tags.py", "test_betteringest_ingest.py"],
    "5": ["test_mcq_precompute.py"],
    "6": ["test_debug_cache.py"],
    "legacy": ["test_pageindex_units.py", "test_chat_endpoint_units.py", "test_retrieval.py"],
}


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", action="append", metavar="N",
                    help=f"limit to one phase ({', '.join(PHASE_FILES)}); repeatable")
    ap.add_argument("--llm", action="store_true",
                    help="also run the live-Ollama tier (needs a running Ollama)")
    ap.add_argument("--only-llm", action="store_true", help="run ONLY the live-Ollama tier")
    args, passthrough = ap.parse_known_args()

    cmd = [sys.executable, "-m", "pytest"]

    if args.phase:
        for phase in args.phase:
            if phase not in PHASE_FILES:
                print(f"unknown phase '{phase}' — choose from {', '.join(PHASE_FILES)}")
                return 2
            for fname in PHASE_FILES[phase]:
                path = ROOT / "tests" / fname
                if path.exists():
                    cmd.append(str(path))
                else:
                    print(f"  (skipping {fname} — not written yet)")

    if args.only_llm:
        cmd += ["-m", "llm", "-o", "addopts="]
        cmd += ["-v", "-ra", "--tb=short", "--color=yes"]
    elif args.llm:
        cmd += ["-m", "unit or llm", "-o", "addopts="]
        cmd += ["-v", "-ra", "--tb=short", "--color=yes"]

    cmd += passthrough

    print(f"\n$ {' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
