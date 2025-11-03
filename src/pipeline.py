from __future__ import annotations
from typing import Dict, Any
from src.core.context import build_context
def inspect_text(raw: str) -> Dict[str, Any]:
    ctx = build_context(raw)
    return {
        "clean_text": ctx.norm.clean_text,
        "sentences": ctx.norm.sentences,
        "equations": ctx.norm.equations,
        "inequalities": ctx.norm.inequalities,
        "congruences": ctx.norm.congruences,
        "divisibility": ctx.norm.divisibility,
        "intervals": ctx.norm.intervals,
        "structures": ctx.norm.structures,
        "goal": ctx.norm.goal,
        "quantifiers": ctx.norm.quantifiers,
        "function_sig": ctx.norm.function_sig,
        "variables": ctx.norm.variables,
        "domains": ctx.norm.domains,
        "known_constants": sorted(ctx.consts.keys()),
        "known_parameters": sorted(ctx.params.keys()),
        "known_functions": sorted(ctx.functions.keys()),
    }
