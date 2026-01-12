from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional


SYSTEM_SOLVER = (
    "You are Axiom v1, a math reasoning system. "
    "Solve the user's problem carefully and provide a complete, readable solution. "
    "Use correct definitions and justify key steps."
)


def symbols_to_roles(symbol_table: Any) -> Dict[str, List[str]]:
    """
    Convert your SymbolTable into a simple role->names dict.
    This is intentionally defensive: it only uses .all() and .role fields.
    """
    roles: Dict[str, List[str]] = {"constant": [], "parameter": [], "function": [], "variable": [], "unknown": []}
    try:
        items = symbol_table.all().items()
    except Exception:
        return roles

    for name, sinfo in items:
        r = getattr(sinfo, "role", "unknown") or "unknown"
        roles.setdefault(r, []).append(name)

    for k in list(roles.keys()):
        roles[k] = sorted(set(roles[k]))
    return roles


def build_user_prompt(
    raw_problem: str,
    normalized: Optional[Any] = None,
    symbol_table: Optional[Any] = None,
    include_norm: bool = True,
) -> str:
    """
    Build the USER prompt. By default, includes a compact JSON dump of Normalized + symbol roles.
    Turn include_norm off if you want a plain "Problem: ..." prompt.

    Note: This does NOT include solutions. No examples are hardcoded.
    """
    raw_problem = (raw_problem or "").strip()

    if not include_norm or normalized is None or symbol_table is None:
        return f"Problem:\n{raw_problem}\n\nWrite a complete solution."

    norm_dict = asdict(normalized)  # Normalized is a dataclass in your code
    roles = symbols_to_roles(symbol_table)

    payload = {
        "problem": raw_problem,
        "normalized": norm_dict,
        "symbols": roles,
    }

    blob = json.dumps(payload, ensure_ascii=False, indent=2)

    return (
        "Solve the following math problem.\n\n"
        "Here is the structured parse (for your use):\n"
        f"{blob}\n\n"
        "Now write a complete solution."
    )


def build_chat_example(
    raw_problem: str,
    solution: str,
    normalized: Optional[Any] = None,
    symbol_table: Optional[Any] = None,
    include_norm: bool = True,
    system: str = SYSTEM_SOLVER,
) -> Dict[str, Any]:
    """
    Returns a single training example in chat format:
      {"messages": [{"role":"system","content":...}, {"role":"user","content":...}, {"role":"assistant","content":...}]}

    This is compatible with most SFT trainers (TRL SFTTrainer, Axolotl, etc.).
    """
    user = build_user_prompt(
        raw_problem=raw_problem,
        normalized=normalized,
        symbol_table=symbol_table,
        include_norm=include_norm,
    )

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": (solution or "").strip()},
        ]
    }
