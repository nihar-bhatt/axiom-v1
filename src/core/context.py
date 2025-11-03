# Wraps normalized text and symbol information into a single per-problem object (ProblemContext).
# This is what you pass to solvers, policies, or verifiers downstream.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from src.core.normalization import Normalized, normalize_user_input
from src.core.symbols import SymbolTable, build_symbol_table


@dataclass
class ProblemContext:
    """
    Container for all structured information derived from a raw problem.
    - raw_text: original user input
    - norm: normalized, structured view of the text
    - symbols: contextual roles/domains/constraints for symbols
    - task: one of {"prove", "show", "find", "prove nonexistence"} or None
    """
    raw_text: str
    norm: Normalized
    symbols: SymbolTable
    task: Optional[str]

    def summary(self) -> str:
        """Compact plain-text summary for debugging or prompting."""
        return (
            f"Task: {self.task}\n"
            f"Variables: {self.norm.variables}\n"
            f"Domains: {self.norm.domains}\n"
            f"Equations: {self.norm.equations}\n"
            f"Inequalities: {self.norm.inequalities}\n"
            f"Congruences: {self.norm.congruences}\n"
            f"Divisibility: {self.norm.divisibility}\n"
            f"Structures: {self.norm.structures}\n"
            f"Function sig: {self.norm.function_sig}\n"
        )


def build_context(raw: str) -> ProblemContext:
    """
    Top-level API:
    - normalize the raw text,
    - build a symbol table from the normalized sentences,
    - return a ProblemContext.
    """
    norm = normalize_user_input(raw)
    symbols = build_symbol_table(norm.sentences)
    return ProblemContext(raw_text=raw, norm=norm, symbols=symbols, task=norm.goal)
