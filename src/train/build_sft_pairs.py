# src/train/build_sft_pairs.py
from __future__ import annotations

import argparse, json, os
from typing import Dict, Iterable

def make_prompt(rec: Dict) -> str:
    n = rec["normalized"]
    parts = []
    parts.append("You are AXIOM, a mathematical proof assistant.")
    # Problem text
    parts.append("Problem:")
    parts.append(rec["problem"].strip())

    # Normalized summary (light and general; helps the model condition on structure)
    def kv(label, val):
        return f"- {label}: {val}"

    parts.append("Normalized summary:")
    if n.get("goal"):          parts.append(kv("goal", n["goal"]))
    if n.get("function_sig"):  parts.append(kv("function_sig", n["function_sig"]))
    if n.get("variables"):     parts.append(kv("variables", ", ".join(n["variables"][:12])))
    if n.get("domains"):       parts.append(kv("domains", ", ".join(f"{k}∈{v}" for k,v in n["domains"].items())))
    if n.get("equations"):     parts.append(kv("equations", "; ".join(n["equations"][:3])))
    if n.get("inequalities"):  parts.append(kv("inequalities", "; ".join(n["inequalities"][:3])))
    if n.get("congruences"):   parts.append(kv("congruences", "; ".join(n["congruences"][:3])))
    if n.get("divisibility"):  parts.append(kv("divisibility", "; ".join(n["divisibility"][:3])))
    if n.get("structures"):    parts.append(kv("structures", "; ".join(n["structures"][:5])))
    if n.get("index_ranges"):  parts.append(kv("indices", "; ".join(n["index_ranges"][:3])))
    # Optional: weak plan cues (keep short)
    if rec.get("weak_plan"):
        cues = [step["op"] for step in rec["weak_plan"][:6] if "op" in step]
        if cues:
            parts.append(kv("plan_cues", " → ".join(cues)))

    parts.append("Write a correct, self-contained solution. If a lemma is used, state it explicitly.")
    return "\n".join(parts)

def iter_pairs(inp_path: str):
    with open(inp_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            yield {
                "id": rec["id"],
                "prompt": make_prompt(rec),
                "target": rec["solution"].strip()
            }

def main():
    ap = argparse.ArgumentParser(description="Build SFT (prompt, target) pairs from normalized JSONL.")
    ap.add_argument("--in", required=True, help="Input JSONL (from make_jsonl.py)")
    ap.add_argument("--out", required=True, help="Output JSONL of pairs")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for pair in iter_pairs(args.in):
            w.write(json.dumps(pair, ensure_ascii=False) + "\n")
            n += 1
    print(f"[build_sft_pairs] wrote {n} pairs to {args.out}")

if __name__ == "__main__":
    main()
