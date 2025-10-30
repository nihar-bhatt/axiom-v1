from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from src.core.normalization import Normalized, normalize_user_input
from src.core.symbols import SymbolTable, build_symbol_table
@dataclass
class ProblemContext:
    raw_text: str
    norm: Normalized
    symbols: SymbolTable
    task: Optional[str]
    def summary(self) -> str:
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
    norm = normalize_user_input(raw)
    symbols = build_symbol_table(norm.sentences)
    return ProblemContext(raw_text=raw, norm=norm, symbols=symbols, task=norm.goal)
