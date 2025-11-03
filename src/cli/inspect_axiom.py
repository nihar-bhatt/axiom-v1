# CLI entrypoint to inspect Axiom's normalized structure for a raw math prompt.
# Run from the project root:
#   python -m src.cli.inspect_axiom --problem "raw text"
# If --problem is omitted, the program reads from stdin.

from __future__ import annotations
import argparse
import json
import sys

from src.pipeline import inspect_text


def main() -> None:
    """
    Parse command-line arguments, read the problem text (flag or stdin),
    call the inspection pipeline, and pretty-print the resulting JSON
    structure (clean_text, equations, variables, domains, etc.).
    """
    ap = argparse.ArgumentParser(
        description="Inspect Axiom's normalized structure for a math prompt."
    )
    ap.add_argument(
        "--problem",
        help="Raw problem text. If omitted, reads from stdin."
    )
    args = ap.parse_args()

    # If --problem isn't provided, read the whole stdin stream
    raw = args.problem if args.problem is not None else sys.stdin.read()

    # Build inspection summary dictionary
    out = inspect_text(raw)

    # Pretty-print JSON with UTF-8 symbols intact
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
