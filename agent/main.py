"""
Entry point for the API Metrics Report Agent.

Usage:
    # Run in default mode (from config / env):
    python -m agent.main

    # Run once:
    python -m agent.main --mode onetime

    # Run on schedule:
    python -m agent.main --mode recurrent
"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="API Metrics Report Agent")
    parser.add_argument(
        "--mode",
        choices=["onetime", "recurrent"],
        default=None,
        help="Run mode: 'onetime' for a single run, 'recurrent' for scheduled runs. "
        "Overrides the RUN_MODE env var / config.",
    )
    args = parser.parse_args()

    if args.mode:
        os.environ["RUN_MODE"] = args.mode

    from agent.scheduler import run_scheduler

    run_scheduler()


if __name__ == "__main__":
    main()
