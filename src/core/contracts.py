from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from src.core.actions import Action
@dataclass(frozen=True)
class Problem:
    id: str
    text: str
    topic: Optional[str] = None
@dataclass
class ProofState:
    goals: List[str]
    ctx: Dict[str, Any]
    backend: Dict[str, Any]
    def is_closed(self) -> bool:
        return len(self.goals) == 0
@dataclass(frozen=True)
class StepOutcome:
    ok: bool
    new_state: Optional[ProofState]
    cert: Optional[str] = None
    message: Optional[str] = None
@dataclass(frozen=True)
class ScoredAction:
    action: Action
    logp: float
    prior: float
    value_est: float
@runtime_checkable
class Verifier(Protocol):
    def check_step(self, state: ProofState, action: Action) -> StepOutcome: ...
    def check_trace(self, init_state: ProofState, actions: List[Action]) -> StepOutcome: ...
@runtime_checkable
class Backend(Protocol):
    def init(self, problem: Problem) -> ProofState: ...
    def apply(self, state: ProofState, action: Action) -> StepOutcome: ...
    def render(self, state: ProofState, trace: List[Action]) -> str: ...
    def kernel_check(self, rendered: str) -> Dict[str, Any]: ...
