# src/data/make_jsonl.py
from __future__ import annotations

import argparse, csv, json, hashlib, os
from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from src.core.normalization import normalize_user_input, sentence_split

# --------- minimal loaders ----------

def iter_csv(path: str, problem_field: str, solution_field: str, limit: Optional[int]) -> Iterator[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        n = 0
        for row in reader:
            if problem_field not in row or solution_field not in row:
                continue
            yield row[problem_field].strip(), row[solution_field].strip()
            n += 1
            if limit is not None and n >= limit:
                break

def iter_jsonl(path: str, problem_field: str, solution_field: str, limit: Optional[int]) -> Iterator[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        n = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if problem_field not in obj or solution_field not in obj:
                continue
            yield str(obj[problem_field]).strip(), str(obj[solution_field]).strip()
            n += 1
            if limit is not None and n >= limit:
                break

def choose_iter(fmt: str):
    fmt = fmt.lower()
    if fmt == "csv":
        return iter_csv
    if fmt == "jsonl":
        return iter_jsonl
    raise ValueError(f"Unsupported --format '{fmt}'. Use 'csv' or 'jsonl'.")

# --------- weak plan (optional) ----------

CUE_MAP = [
    ("Lemma",       "ApplyLemma"),
    ("Claim",       "ApplyLemma"),
    ("Fact",        "ApplyLemma"),
    ("Observation", "ApplyLemma"),
    ("Assume",      "Introduce"),
    ("Suppose",     "Introduce"),
    ("Let",         "Introduce"),
    ("Define",      "Introduce"),
    ("Case",        "CaseStart"),
    ("WLOG",        "CaseStart"),
    ("Without loss", "CaseStart"),
    ("Therefore",   "Conclude"),
    ("Thus",        "Conclude"),
    ("Hence",       "Conclude"),
    ("So",          "Conclude"),
    ("Contradiction", "Conclude"),
    ("Contradicts",   "Conclude"),
    ("It suffices",    "Introduce"),
    ("We show",        "Introduce"),
]

def extract_weak_plan(solution_text: str) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    for sent in sentence_split(solution_text):
        s = sent.strip()
        if not s:
            continue
        hit = None
        for cue, op in CUE_MAP:
            if s.lower().startswith(cue.lower()):
                hit = (op, s); break
        if hit is None:
            for cue, op in (("Lemma", "ApplyLemma"), ("Claim", "ApplyLemma")):
                if cue.lower() in s.lower():
                    hit = (op, s); break
        if hit:
            op, text = hit
            steps.append({"op": op, "text": text})
    if len(steps) > 10:
        steps = steps[:10]
    return steps

# --------- record building ----------

def _hash_id(source: str, problem: str, solution: str) -> str:
    h = hashlib.sha1()
    h.update(source.encode("utf-8")); h.update(b"\x1f")
    h.update(problem.encode("utf-8")); h.update(b"\x1f")
    h.update(solution.encode("utf-8"))
    return h.hexdigest()[:16]

def dataclass_to_dict(x):
    return asdict(x) if is_dataclass(x) else x

def build_record(
    uid: str,
    source: str,
    split: str,
    problem: str,
    solution: str,
    add_plan: bool = True,
) -> Dict:
    norm = normalize_user_input(problem)
    rec = {
        "id": uid,
        "source": source,
        "split": split,
        "problem": problem,
        "solution": solution,
        "normalized": dataclass_to_dict(norm),
        "retrieved_lemmas": [],
        "meta": {
            "has_equation": bool(norm.equations),
            "vars": norm.variables,
            "domains": norm.domains,
        }
    }
    if add_plan:
        rec["weak_plan"] = extract_weak_plan(solution)
    return rec

def write_jsonl(out_path: str, records: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"[make_jsonl] wrote {n} records to {out_path}")

# --------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Build training JSONL from a raw dataset using Axiom normalization."
    )
    ap.add_argument("--in", dest="inp", required=True, help="Path to CSV or JSONL dataset")
    ap.add_argument("--format", choices=["csv", "jsonl"], required=True, help="Input file format")
    ap.add_argument("--problem-field", default="problem", help="Problem column/key name")
    ap.add_argument("--solution-field", default="solution", help="Solution column/key name")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--source", required=True, help="Dataset tag, e.g., aops-nt-2023")
    ap.add_argument("--split", default="train", choices=["train","val","test"], help="Split label")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of examples")
    ap.add_argument("--no-plan", action="store_true", help="Disable weak plan extraction")
    args = ap.parse_args()

    loader = choose_iter(args.format)

    def gen():
        for problem, solution in loader(args.inp, args.problem_field, args.solution_field, args.limit):
            uid = _hash_id(args.source, problem, solution)
            yield build_record(
                uid=uid,
                source=args.source,
                split=args.split,
                problem=problem,
                solution=solution,
                add_plan=not args.no_plan,
            )

    write_jsonl(args.out, gen())

if __name__ == "__main__":
    main()
