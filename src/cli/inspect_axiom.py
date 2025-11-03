from __future__ import annotations
import argparse, json, sys
from src.pipeline import inspect_text
def main():
    ap = argparse.ArgumentParser(description="Inspect Axiom's normalized structure for a math prompt.")
    ap.add_argument("--problem", help="Raw problem text. If omitted, reads from stdin.")
    args = ap.parse_args()
    raw = args.problem if args.problem is not None else sys.stdin.read()
    out = inspect_text(raw)
    print(json.dumps(out, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
