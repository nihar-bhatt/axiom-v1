from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from tqdm import tqdm
from datasets import load_dataset

from src.core.context import build_context
from src.train.prompts import build_chat_example


PROBLEM_KEYS = ["problem", "question", "prompt", "instruction", "input"]
SOLUTION_KEYS = ["solution", "answer", "output", "response", "completion"]


def _first_present(d: Dict[str, Any], keys: list[str]) -> Optional[str]:
    lower = {k.lower(): k for k in d.keys()}
    for want in keys:
        k = lower.get(want.lower())
        if k is not None:
            return k
    return None


def extract_problem_solution(example: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Robust extractor:
    1) Try common direct columns (problem/solution, instruction/output, etc.)
    2) Try "conversations"/"messages" style datasets if present.

    Returns (problem, solution) or None if it can't extract a pair.
    """
    pk = _first_present(example, PROBLEM_KEYS)
    sk = _first_present(example, SOLUTION_KEYS)
    if pk and sk and isinstance(example.get(pk), str) and isinstance(example.get(sk), str):
        p = example[pk].strip()
        s = example[sk].strip()
        if p and s:
            return p, s

    msgs = example.get("messages")
    if isinstance(msgs, list) and msgs:
        user = None
        assistant = None
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").lower()
            content = m.get("content")
            if role == "user" and isinstance(content, str) and user is None:
                user = content.strip()
            if role == "assistant" and isinstance(content, str) and assistant is None:
                assistant = content.strip()
        if user and assistant:
            return user, assistant

    conv = example.get("conversations")
    if isinstance(conv, list) and conv:
        user = None
        assistant = None
        for m in conv:
            if not isinstance(m, dict):
                continue
            frm = (m.get("from") or "").lower()
            val = m.get("value")
            if frm in {"human", "user"} and isinstance(val, str) and user is None:
                user = val.strip()
            if frm in {"gpt", "assistant"} and isinstance(val, str) and assistant is None:
                assistant = val.strip()
        if user and assistant:
            return user, assistant

    return None


def iter_examples(ds) -> Iterator[Dict[str, Any]]:
    # datasets Dataset is iterable; streaming Dataset too.
    for ex in ds:
        if isinstance(ex, dict):
            yield ex


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a JSONL training corpus from AoPS-Instruct (no hardcoded examples).")
    ap.add_argument("--dataset", default="DeepStudentLlama/AoPS-Instruct")
    ap.add_argument("--config", default="default")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="data/processed/aops_sft.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    ap.add_argument("--include_norm", action="store_true", help="Include normalized parse + symbol roles in prompt.")
    ap.add_argument("--streaming", action="store_true", help="Use streaming=True (good for huge datasets).")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=args.streaming)

    total = args.limit if args.limit and args.limit > 0 else None

    written = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        it = iter_examples(ds)
        for ex in tqdm(it, total=total, desc="building-jsonl"):
            if total is not None and (written + skipped) >= total:
                break

            pair = extract_problem_solution(ex)
            if not pair:
                skipped += 1
                continue

            problem, solution = pair

            ctx = build_context(problem)

            item = build_chat_example(
                raw_problem=problem,
                solution=solution,
                normalized=ctx.norm,
                symbol_table=ctx.symbols,
                include_norm=args.include_norm,
            )

            item["meta"] = {
                "source_dataset": args.dataset,
                "split": args.split,
                "include_norm": bool(args.include_norm),
            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nWrote {written} examples to {out_path}")
    print(f"Skipped {skipped} rows (couldn't extract problem/solution)")


if __name__ == "__main__":
    main()
