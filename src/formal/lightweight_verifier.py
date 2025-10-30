from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import sympy as sp
from sympy.core.sympify import SympifyError
from src.core.actions import Action
from src.core.contracts import ProofState, StepOutcome
_EQ_SEPARATORS: Tuple[str, ...] = ("==", "=", "≡")
def _normalize_expr_str(s: str) -> str:
    s = s.strip().rstrip(".")
    s = s.replace("^", "**")
    return s
def _split_equality(expr_text: str) -> Optional[Tuple[str, str]]:
    for sep in _EQ_SEPARATORS:
        if sep in expr_text:
            lhs, rhs = expr_text.split(sep, 1)
            return lhs.strip(), rhs.strip()
    return None
def _equivalent_zero(lhs: sp.Expr, rhs: sp.Expr) -> bool:
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
    try:
        if action.op in {"Rewrite", "Compute"}:
            expr_text = action.args.get("expr", "")
            if not expr_text:
                return StepOutcome(False, state, message="missing expr")
            expr_text = _normalize_expr_str(expr_text)
            split = _split_equality(expr_text)
            if not split:
                return StepOutcome(False, state, message="no equality found")
            left_s, right_s = split
            try:
                lhs = sp.sympify(left_s)
                rhs = sp.sympify(right_s)
            except SympifyError:
                return StepOutcome(False, state, message="sympify failed")
            ok = _equivalent_zero(lhs, rhs)
            return StepOutcome(ok, state, cert="algebraic" if ok else None)
        elif action.op in {"Conclude", "Assume", "Introduce"}:
            return StepOutcome(True, state)
        else:
            return StepOutcome(False, state, message=f"unsupported op {action.op}")
    except Exception as e:
        return StepOutcome(False, state, message=f"exception: {e}")
def check_trace(init_state: ProofState, actions: List[Action]) -> StepOutcome:
    state = init_state
    for a in actions:
        out = check_step(state, a)
        if not out.ok:
            return out
        state = out.new_state or state
    return StepOutcome(True, state, cert="sequence-ok")
