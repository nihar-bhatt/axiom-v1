# Lightweight algebraic verifier for single steps and sequences.
# - Verifies "Rewrite" and "Compute" steps by checking symbolic equality with SymPy.
# - Accepts "Introduce", "Assume", and "Conclude" as structurally valid (no math check).
# - Returns StepOutcome for each step, and aggregates over traces.

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import sympy as sp
from sympy.core.sympify import SympifyError

from src.core.actions import Action
from src.core.contracts import ProofState, StepOutcome

# Allowed separators for equalities. We keep congruence '≡' in the list
# only to normalize inputs; we still treat it as an equality check here.
_EQ_SEPARATORS: Tuple[str, ...] = ("==", "=", "≡")


def _normalize_expr_str(s: str) -> str:
    """
    Normalize a raw math string before parsing:
    - strip surrounding whitespace and trailing periods
    - map '^' to '**' for SymPy exponent syntax
    """
    s = s.strip().rstrip(".")
    s = s.replace("^", "**")
    return s


def _split_equality(expr_text: str) -> Optional[Tuple[str, str]]:
    """
    Split an equality "lhs = rhs" on the first matching separator.
    Returns (lhs, rhs) or None if no equality-like separator is present.
    """
    for sep in _EQ_SEPARATORS:
        if sep in expr_text:
            lhs, rhs = expr_text.split(sep, 1)
            return lhs.strip(), rhs.strip()
    return None


def _equivalent_zero(lhs: sp.Expr, rhs: sp.Expr) -> bool:
    """
    Decide if lhs == rhs symbolically.
    Strategy:
    1) Use SymPy's .equals(0) for strong semantic check of (lhs - rhs) == 0.
    2) If undecided (None), fall back to simplify(lhs - rhs) == 0.
    """
    truth = (lhs - rhs).equals(0)
    if truth is True:
        return True
    if truth is False:
        return False
    try:
        return sp.simplify(lhs - rhs) == 0
    except Exception:
        return False


def check_step(state: ProofState, action: Action) -> StepOutcome:
    """
    Verify a single action against the current proof state.
    Supported ops:
      - Rewrite/Compute: parse and check symbolic equality
      - Introduce/Assume/Conclude: accepted structurally (no math check)
    Returns a StepOutcome with ok flag and optional message/cert.
    """
    try:
        if action.op in {"Rewrite", "Compute"}:
            expr_text = action.args.get("expr", "")
            if not expr_text:
                return StepOutcome(False, state, message="missing expr")

            # Normalize and split into sides
            expr_text = _normalize_expr_str(expr_text)
            split = _split_equality(expr_text)
            if not split:
                return StepOutcome(False, state, message="no equality found")
            left_s, right_s = split

            # Parse into SymPy expressions
            try:
                lhs = sp.sympify(left_s)
                rhs = sp.sympify(right_s)
            except SympifyError:
                return StepOutcome(False, state, message="sympify failed")

            # Decide equality
            ok = _equivalent_zero(lhs, rhs)
            return StepOutcome(ok, state, cert="algebraic" if ok else None)

        elif action.op in {"Conclude", "Assume", "Introduce"}:
            # These are accepted as structural steps; no algebra check needed
            return StepOutcome(True, state)

        else:
            # Unrecognized operation — not verified by this lightweight checker
            return StepOutcome(False, state, message=f"unsupported op {action.op}")

    except Exception as e:
        # Guard against unexpected parsing/compute errors to keep the pipeline robust
        return StepOutcome(False, state, message=f"exception: {e}")


def check_trace(init_state: ProofState, actions: List[Action]) -> StepOutcome:
    """
    Verify a full sequence of actions.
    Evaluates each step in order, returning the first failure or success for all.
    """
    state = init_state
    for a in actions:
        out = check_step(state, a)
        if not out.ok:
            return out
        # If a backend ever returns a new_state, propagate it; otherwise keep state
        state = out.new_state or state
    return StepOutcome(True, state, cert="sequence-ok")
