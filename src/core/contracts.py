# Shared data contracts and interfaces used by search, verification, and backends.
# This file declares light-weight dataclasses and Protocols; no heavy logic here.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable

from src.core.actions import Action


@dataclass(frozen=True)
class Problem:
    """Minimal problem metadata for a backend or search loop."""
    id: str
    text: str
    topic: Optional[str] = None


@dataclass
class ProofState:
    """
    Backend-facing state:
    - goals: human-readable goals or backend-rendered subgoals
    - ctx:   local assumptions, definitions, or symbol bindings
    - backend: opaque handles (e.g., Lean REPL IDs)
    """
    goals: List[str]
    ctx: Dict[str, Any]
    backend: Dict[str, Any]

    def is_closed(self) -> bool:
        """Return True if there are no remaining goals."""
        return len(self.goals) == 0


@dataclass(frozen=True)
class StepOutcome:
    """
    Result of checking or applying an action:
    - ok: whether the action succeeded
    - new_state: the next ProofState (or None if unchanged)
    - cert/message: optional certificate or diagnostic note
    """
    ok: bool
    new_state: Optional[ProofState]
    cert: Optional[str] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class ScoredAction:
    """Action proposal with policy/value scores (for search)."""
    action: Action
    logp: float
    prior: float
    value_est: float


@runtime_checkable
class Verifier(Protocol):
    """Verifier interface for checking single steps or full traces."""
    def check_step(self, state: ProofState, action: Action) -> StepOutcome: ...
    def check_trace(self, init_state: ProofState, actions: List[Action]) -> StepOutcome: ...


@runtime_checkable
class Backend(Protocol):
    """
    Backend interface (e.g., Lean REPL).
    - init: construct initial state
    - apply: apply an action to a state
    - render: produce a final file/string for kernel checking
    - kernel_check: run the checker and return diagnostics
    """
    def init(self, problem: Problem) -> ProofState: ...
    def apply(self, state: ProofState, action: Action) -> StepOutcome: ...
    def render(self, state: ProofState, trace: List[Action]) -> str: ...
    def kernel_check(self, rendered: str) -> Dict[str, Any]: ...
