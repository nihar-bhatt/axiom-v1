# Defines the canonical "Action" representation for proof steps.
# Each action has:
#   - op:       the operation name (Introduce, Rewrite, Compute, Conclude, ...)
#   - args:     a small dict of arguments (e.g., {"expr": "x = 2n"})
#   - step_id:  optional step identifier ("s1", "s2", ...)
#   - depends_on: list of previous step_ids that this step requires
#
# The helper functions validate sequences and convert to/from JSON.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

# Operation schema: op -> [(arg_name, type_str, required_bool)]
ACTION_SCHEMA: Dict[str, List[Tuple[str, str, bool]]] = {
    "Introduce":   [("text", "str", True)],
    "Assume":      [("text", "str", True)],
    "Recall":      [("lemma", "str", True)],
    "ApplyLemma":  [("lemma", "str", True), ("premises", "str", False)],
    "Rewrite":     [("expr", "str", True), ("rule", "str", False)],
    "Compute":     [("expr", "str", True)],
    "Derive":      [("text", "str", True)],
    "CaseStart":   [("condition", "str", True)],
    "CaseEnd":     [("summary", "str", False)],
    "Conclude":    [("text", "str", True)],
}


@dataclass
class Action:
    """
    Canonical proof-step object used across the system.
    Validated against ACTION_SCHEMA.
    """
    op: str
    args: Dict[str, Any] = field(default_factory=dict)
    step_id: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)

    def validate(self) -> None:
        """Check that the op is known and all required args have correct types."""
        if self.op not in ACTION_SCHEMA:
            raise ValueError(f"Unknown action op: {self.op}")

        schema = ACTION_SCHEMA[self.op]
        required = {k for k, _, req in schema if req}
        allowed = {k for k, _, _ in schema}

        # Required args must be present
        missing = [k for k in required if k not in self.args]
        if missing:
            raise ValueError(f"Action '{self.op}' missing required args: {missing}")

        # No unexpected args
        extra = set(self.args) - allowed
        if extra:
            raise ValueError(f"Action '{self.op}' has unknown args: {sorted(extra)}")

        # Basic type checks for string args
        for k, typ, _ in schema:
            if k in self.args:
                v = self.args[k]
                if typ == "str" and not isinstance(v, str):
                    raise TypeError(f"Action '{self.op}' arg '{k}' must be str")

    def to_tokens(self) -> List[str]:
        """
        Serialize to a simple token sequence for a sequence model, e.g.:
        ["<Rewrite>", "<expr>", "x = 2n", "</step>"].
        """
        toks = [f"<{self.op}>"]
        for k, _, _ in ACTION_SCHEMA[self.op]:
            if k in self.args:
                toks += [f"<{k}>", self.args[k]]
        toks.append("</step>")
        return toks

    @staticmethod
    def from_tokens(toks: List[str]) -> "Action":
        """
        Parse a token sequence produced by to_tokens() back into an Action.
        """
        if not toks or not toks[0].startswith("<") or toks[-1] != "</step>":
            raise ValueError("Invalid token sequence")

        op = toks[0].strip("<>")
        args: Dict[str, Any] = {}

        i = 1
        while i < len(toks) - 1:
            if toks[i].startswith("<") and toks[i].endswith(">"):
                key = toks[i].strip("<>")
                i += 1
                if i >= len(toks) - 1:
                    raise ValueError("Missing value for arg")
                args[key] = toks[i]
                i += 1
            else:
                i += 1

        a = Action(op=op, args=args)
        a.validate()
        return a


def validate_sequence(seq: List[Action]) -> None:
    """
    Validate the full sequence:
    - assign step_ids if missing
    - each step is valid
    - dependency edges only point to prior steps
    """
    seen: set[str] = set()
    for idx, a in enumerate(seq, 1):
        if not a.step_id:
            a.step_id = f"s{idx}"
        a.validate()
        for dep in a.depends_on:
            if dep not in seen:
                raise ValueError(f"{a.step_id} depends on unknown or future step '{dep}'")
        seen.add(a.step_id)


def actions_to_json(seq: List[Action]) -> List[Dict[str, Any]]:
    """Convert a validated sequence to a JSON-serializable list of dicts."""
    validate_sequence(seq)
    return [
        {"step_id": a.step_id, "op": a.op, "args": a.args, "depends_on": a.depends_on}
        for a in seq
    ]


def json_to_actions(items: List[Dict[str, Any]]) -> List[Action]:
    """Reconstruct and validate a sequence of Actions from a JSON list."""
    seq: List[Action] = []
    for d in items:
        a = Action(
            op=d["op"],
            args=d.get("args", {}),
            step_id=d.get("step_id"),
            depends_on=d.get("depends_on", []),
        )
        a.validate()
        seq.append(a)
    validate_sequence(seq)
    return seq
