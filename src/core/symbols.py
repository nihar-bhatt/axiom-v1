from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
import re
Role = Literal["variable", "parameter", "constant", "function", "unknown"]
Domain = str
BUILTIN_CONSTANTS = {
    "π": {"lean": "Real.pi", "sympy": "pi"},
    "pi": {"lean": "Real.pi", "sympy": "pi"},
    "e":  {"lean": "Real.exp 1", "sympy": "E"},
    "i":  {"lean": "Complex.I", "sympy": "I"},
    "I":  {"lean": "Complex.I", "sympy": "I"},
}
PAT_CONST = [
    re.compile(r"\b(let|fix)\s+([A-Za-z])\b.*\bconstant\b", re.I),
    re.compile(r"\bfor\s+some\s+constant\s+([A-Za-z])\b", re.I),
    re.compile(r"\b([A-Za-z])\s+is\s+a\s+constant\b", re.I),
]
PAT_PARAM = [
    re.compile(r"\bfor\s+all\s+([A-Za-z](?:\s*,\s*[A-Za-z])*)\b", re.I),
    re.compile(r"\bfor\s+any\s+([A-Za-z](?:\s*,\s*[A-Za-z])*)\b", re.I),
    re.compile(r"\bfor\s+every\s+([A-Za-z](?:\s*,\s*[A-Za-z])*)\b", re.I),
    re.compile(r"\bthere\s+exists\s+([A-Za-z])\b", re.I),
    re.compile(r"\bfix\s+([A-Za-z])\b", re.I),
]
PAT_FUNC_SIG = [
    re.compile(r"\b([A-Za-z])\s*[:∶]\s*([^\s-]+)\s*[-–—]?>\s*([^\s.,;]+)")
]
PAT_PRIME = re.compile(r"\b([A-Za-z])\s+is\s+prime\b|\bprime\s+([A-Za-z])\b", re.I)
PAT_SIGN  = re.compile(r"\b([A-Za-z])\s*>\s*0\b|\b([A-Za-z])\s*≥\s*0\b|\b([A-Za-z])\s*>=\s*0\b")
PAT_DOMAIN_IN = re.compile(r"\b([A-Za-z](?:\s*,\s*[A-Za-z])*)\s*(?:∈|in)\s+([A-Za-z_][^,.;\s]*)")
@dataclass
class Evidence:
    source: str
    kind: str
    weight: float = 1.0
@dataclass
class SymbolInfo:
    name: str
    role: Role = "unknown"
    domain: Optional[Domain] = None
    constraints: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    confidence: float = 0.0
    def add_evidence(self, ev: Evidence) -> None:
        self.evidence.append(ev); self._recompute_confidence()
    def add_constraint(self, c: str) -> None:
        if c not in self.constraints: self.constraints.append(c)
    def set_domain(self, d: Domain, src: str, w: float = 0.75) -> None:
        self.domain = d; self.add_evidence(Evidence(src, "domain", w))
    def set_role(self, r: Role, src: str, w: float = 0.75) -> None:
        self.role = r; self.add_evidence(Evidence(src, "decl", w))
    def _recompute_confidence(self) -> None:
        score = 0.0
        for e in self.evidence:
            score += {"builtin":1.0, "decl":0.7, "quantifier":0.6,
                      "domain":0.6, "constraint":0.5, "funcsig":0.5}.get(e.kind, 0.3)*e.weight
        self.confidence = min(1.0, score / (1.0 + 0.2*len(self.evidence)))
class SymbolTable:
    def __init__(self): self._tbl: Dict[str, SymbolInfo] = {}
    def ensure(self, name: str) -> SymbolInfo:
        if name not in self._tbl: self._tbl[name] = SymbolInfo(name=name)
        return self._tbl[name]
    def get(self, name: str) -> Optional[SymbolInfo]: return self._tbl.get(name)
    def all(self) -> Dict[str, SymbolInfo]: return self._tbl
    def seed_builtins(self, text: str) -> None:
        for k in BUILTIN_CONSTANTS:
            if re.search(rf"\b{k}\b", text):
                si = self.ensure(k); si.set_role("constant", "builtin", 1.0)
                si.add_evidence(Evidence("builtin", "builtin", 1.0))
    def scan_quantifiers(self, sent: str) -> None:
        for pat in PAT_PARAM:
            for m in pat.finditer(sent):
                for v in [x.strip() for x in m.group(1).split(",") if x.strip()]:
                    si = self.ensure(v)
                    if si.role == "unknown": si.set_role("parameter", "quantifier", 0.6)
                    si.add_evidence(Evidence(sent, "quantifier", 0.6))
    def scan_constants(self, sent: str) -> None:
        for pat in PAT_CONST:
            for m in pat.finditer(sent):
                v = m.group(2) if m.lastindex and m.lastindex >= 2 else None
                if v: self.ensure(v).set_role("constant", "declaration", 0.8)
        for m in PAT_SIGN.finditer(sent):
            for g in m.groups():
                if g: self.ensure(g).add_constraint(">0")
        pm = PAT_PRIME.search(sent)
        if pm:
            v = pm.group(1) or pm.group(2)
            if v:
                si = self.ensure(v); si.add_constraint("prime")
                if si.role == "unknown": si.set_role("parameter", "prime-mention", 0.6)
    def scan_domains(self, sent: str) -> None:
        for m in PAT_DOMAIN_IN.finditer(sent):
            vs = [x.strip() for x in m.group(1).split(",")]
            raw_dom = m.group(2).strip()
            for v in vs:
                if v: self.ensure(v).set_domain(raw_dom, sent, 0.7)
    def scan_function_sigs(self, sent: str) -> None:
        for pat in PAT_FUNC_SIG:
            for m in pat.finditer(sent.replace(" ", "")):
                f = m.group(1)
                si = self.ensure(f); si.set_role("function", "funcsig", 0.8)
                si.add_evidence(Evidence(sent, "funcsig", 0.8))
    def finalize(self) -> Dict[str, SymbolInfo]:
        for name, si in self._tbl.items():
            if si.role == "unknown":
                if si.constraints and not any(ev.kind == "quantifier" for ev in si.evidence):
                    si.set_role("constant", "late-constraint", 0.4)
                if si.role == "unknown": si.set_role("variable", "default", 0.2)
        return self._tbl
def build_symbol_table(sentences: List[str]) -> SymbolTable:
    st = SymbolTable()
    joined = " ".join(sentences)
    st.seed_builtins(joined)
    for s in sentences:
        st.scan_quantifiers(s)
        st.scan_constants(s)
        st.scan_domains(s)
        st.scan_function_sigs(s)
    st.finalize()
    return st
