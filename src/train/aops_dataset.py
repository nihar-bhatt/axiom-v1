from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from datasets import load_dataset

from src.core.context import build_context

def load_aops(split: str = "train", config: str = "default"):
    return load_dataset("DeepStudentLlama/AoPS-Instruct", config, split=split)

def extract_xy(ex: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    x = ex.get("problem") or ex.get("question") or ex.get("prompt")
    y = ex.get("solution") or ex.get("answer") or ex.get("output")
    if not isinstance(x, str) or not isinstance(y, str):
        return None
    x, y = x.strip(), y.strip()
    if not x or not y:
        return None
    return x, y

def build_prompt(problem: str, include_norm: bool = True) -> str:
    if not include_norm:
        return f"Problem:\n{problem}\n\nWrite a complete solution."

    ctx = build_context(problem)
    norm = ctx.norm

    return (
        "Problem:\n"
        f"{problem.strip()}\n\n"
        "Auto-extracted structure (may be incomplete):\n"
        f"- Goal: {norm.goal}\n"
        f"- Function signature: {norm.function_sig}\n"
        f"- Variables: {norm.variables}\n"
        f"- Domains: {norm.domains}\n"
        f"- Equations: {norm.equations}\n"
        f"- Inequalities: {norm.inequalities}\n"
        f"- Congruences: {norm.congruences}\n"
        f"- Divisibility: {norm.divisibility}\n"
        f"- Structures/Tags: {norm.structures}\n\n"
        "Task: Write a complete, correct solution."
    )
