# High-level inspection pipeline:
# - Build a ProblemContext from raw text
# - Group symbols by role (constant/parameter/function/variable/unknown)
# - Return a JSON-serializable summary of the normalized structure

from __future__ import annotations
from typing import Dict, Any

from src.core.context import build_context


def inspect_text(raw: str) -> Dict[str, Any]:
    """
    Convert raw, possibly noisy mathematical text into a structured summary.

    Returns a dictionary containing:
      - normalized text and sentences
      - extracted structures: equations, inequalities, congruences, divisibility, intervals
      - detected meta: goal, quantifiers, function signature
      - symbol information: variables/domains, grouped symbol roles

    This function is used by the CLI to print a JSON preview of what Axiom
    understood from the input.
    """
    ctx = build_context(raw)

    # Derive symbol groups from the symbol table
    roles = {"constant": [], "parameter": [], "function": [], "variable": [], "unknown": []}
    for name, sinfo in ctx.symbols.all().items():
        roles.setdefault(sinfo.role, []).append(name)

    # Sort for stable output
    for k in roles:
        roles[k] = sorted(set(roles[k]))

    return {
        # Raw → clean text and basic structure
        "clean_text":   ctx.norm.clean_text,
        "sentences":    ctx.norm.sentences,
        "equations":    ctx.norm.equations,
        "inequalities": ctx.norm.inequalities,
        "congruences":  ctx.norm.congruences,
        "divisibility": ctx.norm.divisibility,
        "intervals":    ctx.norm.intervals,
        "structures":   ctx.norm.structures,

        # Tasks and signatures
        "goal":         ctx.norm.goal,
        "quantifiers":  ctx.norm.quantifiers,
        "function_sig": ctx.norm.function_sig,

        # Symbols: variables, domains, and grouped roles
        "variables":        ctx.norm.variables,
        "domains":          ctx.norm.domains,
        "known_constants":  roles["constant"],
        "known_parameters": roles["parameter"],
        "known_functions":  roles["function"],
        "known_variables":  roles["variable"],
        "unknown_symbols":  roles["unknown"],
    }
