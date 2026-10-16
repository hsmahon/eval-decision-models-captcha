#!/usr/bin/env python3
"""Evaluate pinned Perplexity decision models through OpenRouter.

Uses the same crops, prompt, criteria, request format, metrics, and resumable
checkpoints as the Jev benchmark. Run each model separately to resume its run.
"""

import argparse
import sys

import eval_jev as benchmark

MODELS = {
    "pplx-decider-flash-v1.2": 0.01,
    "pplx-decider-v1.2": 0.02,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    args, remaining = parser.parse_known_args()
    benchmark.MODEL = f"perplexity/{args.model}"
    benchmark.PRICE_PER_MILLION_INPUT_TOKENS = MODELS[args.model]
    benchmark.RESULT_PATH = benchmark.ROOT / f"data/results_{args.model}.json"
    benchmark.PROGRESS_PATH = benchmark.ROOT / f"data/.{args.model}_progress.json"
    benchmark.BACKEND = "openrouter-perplexity"
    sys.argv = [sys.argv[0], *remaining]
    return benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
