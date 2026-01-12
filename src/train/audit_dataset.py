from __future__ import annotations

import argparse
import statistics as stats
from typing import Any, Dict, Optional, Tuple

from datasets import load_dataset


def _is_string_feature(feat: Any) -> bool:
    """
    HuggingFace features are usually datasets.Value('string') or similar.
    We check via repr/name to avoid importing private feature classes.
    """
    r = repr(feat).lower()
    return ("string" in r) or ("value(dtype='string')" in r)


def _pick_key(column_names: list[str], candidates: list[str]) -> Optional[str]:
    """
    Pick the first candidate present in column_names (case-insensitive).
    """
    lower_map = {c.lower(): c for c in column_names}
    for k in candidates:
        if k.lower() in lower_map:
            return lower_map[k.lower()]
    return None


def infer_problem_solution_keys(ds) -> Tuple[Optional[str], Optional[str]]:
    """
    Infer which columns likely contain (problem, solution) WITHOUT printing any examples.
    Uses dataset feature types + common naming conventions.
    """
    colnames = list(ds.column_names)
    feats = getattr(ds, "features", {}) or {}

    # Only consider columns that look like strings
    string_cols = []
    for c in colnames:
        f = feats.get(c)
        if f is None:
            # If features missing, fall back to name-only heuristics
            string_cols.append(c)
        else:
            if _is_string_feature(f):
                string_cols.append(c)

    # Heuristics: common names in math instruction datasets
    problem_candidates = ["problem", "question", "prompt", "input", "instruction"]
    solution_candidates = ["solution", "answer", "output", "response", "completion"]

    pk = _pick_key(string_cols, problem_candidates)
    sk = _pick_key(string_cols, solution_candidates)

    # If still missing, try a weaker heuristic: pick the two biggest-looking string fields by name
    # (This is intentionally conservative.)
    if pk is None or sk is None:
        # Prefer "instruction"/"prompt" as problem; "output"/"response" as solution
        pk = pk or _pick_key(string_cols, ["instruction", "prompt", "input", "question"])
        sk = sk or _pick_key(string_cols, ["output", "response", "completion", "answer"])

    return pk, sk


def audit_lengths(ds, pk: str, sk: str, limit: int = 2000) -> Dict[str, Any]:
    """
    Collect basic stats for the first `limit` examples:
      - how many usable rows
      - character lengths (min/median/mean/max)
    Does NOT print the actual text.
    """
    p_lens: list[int] = []
    s_lens: list[int] = []
    usable = 0

    for i, ex in enumerate(ds):
        if i >= limit:
            break
        p = ex.get(pk)
        s = ex.get(sk)
        if not isinstance(p, str) or not isinstance(s, str):
            continue
        p = p.strip()
        s = s.strip()
        if not p or not s:
            continue
        usable += 1
        p_lens.append(len(p))
        s_lens.append(len(s))

    def pack(xs: list[int]) -> Dict[str, float]:
        if not xs:
            return {}
        return {
            "min": float(min(xs)),
            "median": float(stats.median(xs)),
            "mean": float(stats.mean(xs)),
            "max": float(max(xs)),
        }

    return {
        "scanned": min(limit, len(ds)),
        "usable": usable,
        "problem_chars": pack(p_lens),
        "solution_chars": pack(s_lens),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit an HF dataset schema + basic stats (no text dumping).")
    ap.add_argument("--dataset", default="DeepStudentLlama/AoPS-Instruct")
    ap.add_argument("--config", default="default")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--problem_key", default=None)
    ap.add_argument("--solution_key", default=None)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.config, split=args.split)

    print("Loaded dataset:", args.dataset)
    print("Config:", args.config, "| Split:", args.split)
    print("Columns:", ds.column_names)

    pk, sk = args.problem_key, args.solution_key
    if pk is None or sk is None:
        gpk, gsk = infer_problem_solution_keys(ds)
        pk = pk or gpk
        sk = sk or gsk

    if not pk or not sk:
        print("\nCould not confidently infer problem/solution keys.")
        print("Run again with --problem_key ... --solution_key ... after you inspect ds.column_names.")
        return

    print("\nUsing keys:")
    print("  problem_key =", pk)
    print("  solution_key =", sk)

    report = audit_lengths(ds, pk, sk, limit=args.limit)
    print("\nBasic stats (no text shown):")
    print(report)


if __name__ == "__main__":
    main()
