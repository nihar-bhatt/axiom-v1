# Normalizes raw mathematical text and extracts structure:
# - cleans LaTeX/Unicode, fixes symbols and spacing
# - detects equations, inequalities, congruences, divisibility, intervals
# - extracts variables, domains, quantifiers, function signatures
# - outputs a Normalized dataclass (JSON-friendly)

from __future__ import annotations
import re
import difflib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# --- canonical symbol handling ------------------------------------------------

REL_MAP = {
    "≠": "!=",
    "≤": "<=", "≥": ">=",
    "→": "->", "⇒": "->", r"\to": "->", r"\rightarrow": "->",
    "⇔": "<->",
    "−": "-", "–": "-", "—": "-",
}
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

WS          = re.compile(r"[ \t\u00A0]+")
MULTISPACE  = re.compile(r"\s{2,}")
PUNCT_FIX   = re.compile(r"\s*([,;:!?])")
SENT_SPLIT  = re.compile(r"(?<=[.!?])\s+|\n+")

LATEX_INLINES = [
    (re.compile(r"\\\[(.*?)\\\]", re.S), r"\1"),
    (re.compile(r"\\\((.*?)\\\)", re.S), r"\1"),
    (re.compile(r"\\mathrm\{([^}]+)\}"), r"\1"),
    (re.compile(r"\\operatorname\{([^}]+)\}"), r"\1"),
    (re.compile(r"\\text\{([^}]+)\}"), r"\1"),
]

# --- domain aliases (broad) ---------------------------------------------------

DOMAIN_MAP: Dict[str, str] = {
    "r":"R","rr":"R","ℝ":"R","real":"R","reals":"R","realnumbers":"R",
    "q":"Q","qq":"Q","ℚ":"Q","rational":"Q","rationals":"Q",
    "z":"Z","zz":"Z","ℤ":"Z","integer":"Z","integers":"Z",
    "n":"N","ℕ":"N","naturals":"N","naturalnumbers":"N",
    "c":"C","cc":"C","ℂ":"C","complex":"C","complexnumbers":"C",
    "p":"P","ℙ":"P","prime":"P","primes":"P",
    "r+":"R+","r>0":"R+","positivereals":"R+",
    "r-":"R-","negativereals":"R-",
    "r>=0":"R>=0","nonnegativereals":"R>=0",
    "z+":"Z+","positiveintegers":"Z+",
    "z-":"Z-","negativeintegers":"Z-",
    "z>=0":"Z>=0","nonnegativeintegers":"Z>=0",
    "n0":"Z>=0","n+":"N+","n>0":"N+",
    "irrational":"Irrational",
    "transcendental":"Transcendental",
    "algebraic":"Algebraic",
    "surreal":"Surreal",
    "hyperreal":"Hyperreal",
    "infinitesimal":"Infinitesimal",
    "p-adic":"Q_p","padic":"Q_p",
    "r^n":"R^n","rr^n":"R^n","zn":"Z^n","cn":"C^n",
    "group":"Group","ring":"Ring","field":"Field",
    "vectorspace":"VectorSpace","matrixspace":"MatrixSpace",
    "hilbertspace":"HilbertSpace","banachspace":"BanachSpace",
    "pi":"π","tau":"τ","phi":"φ","φ":"φ","π":"π",
}

PHRASE_DOMAIN = {
    "integer":"Z","integers":"Z",
    "rational":"Q","rationals":"Q",
    "real":"R","reals":"R",
    "natural":"N","naturals":"N",
    "complex":"C","complex numbers":"C",
    "prime":"P","primes":"P",
}

# --- structural patterns ------------------------------------------------------

EQ_RE    = re.compile(r"(?P<lhs>[^=<>!≡]+)\s*(==|=)\s*(?P<rhs>[^;.,\n]+)")
INEQ_RE  = re.compile(r"(?P<lhs>[^=<>!]+)\s*(<=|>=|<|>)\s*(?P<rhs>[^;.,\n]+)")
CONG_RE  = re.compile(r"(?P<lhs>[^=≡]+)\s*≡\s*(?P<rhs>[^(\n;.,]+)\s*\(mod\s*(?P<mod>[^)]+)\)", re.I)
DIV_RE   = re.compile(r"\b(?P<a>[A-Za-z0-9\)\]]+)\s*\|\s*(?P<b>[A-Za-z0-9\(\[]+)\b")

IN_PATTERNS = [
    re.compile(r"\b([A-Za-z](?:\s*,\s*[A-Za-z])*)\s*∈\s*([A-Za-zℝℤℚℕℂP\+\>\=\d^ _\[\]()/]+)", re.I),
    re.compile(r"\b([A-Za-z](?:\s*,\s*[A-Za-z])*)\s+in\s+([A-Za-zℝℤℚℕℂP\+\>\=\d^ _\[\]()/]+)", re.I),
]

PHRASE_PATTERNS = [
    (re.compile(r"\b([A-Za-z](?:\s*,\s*[A-Za-z])*)\s+are\s+(?P<dom>integers|rationals|reals|naturals|complex|primes)\b", re.I), "are"),
    (re.compile(r"\b([A-Za-z])\s+is\s+(?P<dom>integer|rational|real|natural|complex|prime)\b", re.I), "is"),
    (re.compile(r"\b([A-Za-z])\s+(?P<dom>complex|prime|integer|rational|real|natural)\b", re.I), "bare"),
]

SIG_RE     = re.compile(r"([A-Za-z])\s*[:∶]\s*([^\s-]+)\s*[-–—]?>\s*([^\s.,;]+)")
FORALL_RE  = re.compile(r"\bfor all\b|\bfor any\b|\bfor every\b", re.I)
EXISTS_RE  = re.compile(r"\bthere (exists|is)\b|\bfind\b|\bdetermine\b|\bcompute\b", re.I)
INTERVAL_RE = re.compile(r"([\(\[])\s*([^,\s]+)\s*,\s*([^)\]\s]+)\s*([\)\]])")

STRUCT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bF_?\{?(\d+)\}?\b", re.I), "F_p"),
    (re.compile(r"\bGF\s*\(\s*(\d+)\s*\)", re.I), "F_q"),
    (re.compile(r"\bZ\s*/\s*(\d+)\s*Z\b", re.I), "Z_n"),
    (re.compile(r"\bZ\s*\[\s*i\s*\]\b", re.I), "Z[i]"),
    (re.compile(r"\bZ\s*\[\s*\w\s*\]\b", re.I), "Z[α]"),
    (re.compile(r"\bQ\s*_?\{?p\}?\b", re.I), "Q_p"),
    (re.compile(r"\bR\^(\d+)\b", re.I), "R^n"),
    (re.compile(r"\bC\^(\d+)\b", re.I), "C^n"),
]

CORRECTIONS = [
    (re.compile(r"\bninter\b", re.I), "n ∈ Z"),
    (re.compile(r"\bn int\b", re.I), "n ∈ Z"),
    (re.compile(r"\bprive\b", re.I), "prove"),
    (re.compile(r"\bproove\b", re.I), "prove"),
    (re.compile(r"\bdoinfo\b", re.I), "is an"),
    (re.compile(r"\bdoinf\b", re.I), "is an"),
    (re.compile(r"\bdoin\b", re.I), "is an"),
    (re.compile(r"\biff\b", re.I), "<->"),
]

VOCAB = [
    "prove","show","determine","compute","integer","integers","rational","rationals",
    "real","reals","natural","naturals","complex","prime","primes","mod","congruent",
    "there","exists","for","all","any","every","such","that","where","assume","suppose","let","then"
]

def _fuzzy_fix_token(tok: str) -> str:
    """Return a close vocabulary fix for a single token, or the token itself."""
    cand = difflib.get_close_matches(tok, VOCAB, n=1, cutoff=0.86)
    return cand[0] if cand else tok

# --- output structure ---------------------------------------------------------

@dataclass
class Normalized:
    """
    Machine-interpretable view of raw mathematical text.
    All fields are JSON-friendly.
    """
    clean_text: str
    sentences: List[str]
    equations: List[str]
    inequalities: List[str]
    congruences: List[str]
    divisibility: List[str]
    intervals: List[str]
    structures: List[str]
    goal: Optional[str]
    quantifiers: Optional[str]
    function_sig: Optional[str]
    variables: List[str]
    domains: Dict[str, str]

# --- pipeline helpers ---------------------------------------------------------

def _strip_latex(s: str) -> str:
    """Remove simple LaTeX wrappers and common \mathbb forms."""
    for pat, repl in LATEX_INLINES:
        s = pat.sub(repl, s)
    s = (s.replace(r"\mathbb{R}", "R")
           .replace(r"\mathbb{Z}", "Z")
           .replace(r"\mathbb{Q}", "Q")
           .replace(r"\mathbb{N}", "N")
           .replace(r"\mathbb{C}", "C"))
    s = s.replace("\\cdot", "*")
    return s

def _normalize_symbols(s: str) -> str:
    """Normalize unicode superscripts and relation symbols, map '^' to '**'."""
    s = s.translate(SUPERSCRIPTS)
    for k, v in REL_MAP.items():
        s = s.replace(k, v)
    s = s.replace("^", "**")
    return s

def _clean_ws(s: str) -> str:
    """Collapse extra whitespace and normalize punctuation spacing."""
    s = WS.sub(" ", s)
    s = re.sub(PUNCT_FIX, r"\1 ", s)
    s = MULTISPACE.sub(" ", s).strip()
    return s

def sentence_split(text: str) -> List[str]:
    """Split into sentences on terminal punctuation or newlines."""
    parts = re.split(SENT_SPLIT, text.strip())
    return [p.strip() for p in parts if p.strip()]

def _infer_goal(text: str) -> Optional[str]:
    """Detect top-level task intent: prove / show / find / nonexistence."""
    tl = text.lower()
    if "prove that" in tl or tl.startswith("prove ") or " prove " in tl: return "prove"
    if "show that" in tl or tl.startswith("show "): return "show"
    if "find " in tl or "determine " in tl or "compute " in tl: return "find"
    if "no such" in tl or "does not exist" in tl or "nonexist" in tl: return "prove nonexistence"
    return None

def _function_signature(text: str) -> Optional[str]:
    """Extract function signature f:A->B when present."""
    m = SIG_RE.search(text.replace(" ", ""))
    if m: return f"{m.group(1)}: {m.group(2)} -> {m.group(3)}"
    return None

def _dedupe(xs: List[str]) -> List[str]:
    """Preserve order while removing duplicates."""
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

# --- extract structural items -------------------------------------------------

def _extract_equations(text: str) -> List[str]:
    return [f"{m.group('lhs').strip()} = {m.group('rhs').strip()}" for m in EQ_RE.finditer(text)]

def _extract_inequalities(text: str) -> List[str]:
    return [m.group(0).strip() for m in INEQ_RE.finditer(text)]

def _extract_congruences(text: str) -> List[str]:
    return [f"{m.group('lhs').strip()} ≡ {m.group('rhs').strip()} (mod {m.group('mod').strip()})"
            for m in CONG_RE.finditer(text)]

def _extract_divisibility(text: str) -> List[str]:
    return [f"{m.group('a')} | {m.group('b')}" for m in DIV_RE.finditer(text)]

def _extract_intervals(text: str) -> List[str]:
    out = []
    for m in INTERVAL_RE.finditer(text):
        l, a, b, r = m.groups()
        out.append(f"{l}{a},{b}{r}")
    return out

def _detect_structures(text: str) -> List[str]:
    """Detect mentions of standard algebraic/number-theoretic structures."""
    found = []
    for pat, tag in STRUCT_PATTERNS:
        for _ in pat.finditer(text):
            found.append(tag)
    return _dedupe(found)

# Variable detection: allow x, x1, a_n, x' but ignore common keywords
KEYWORDS = set("""
prove show determine compute integer integers rational rationals real reals
natural naturals complex primes prime such that where and or iff if then let suppose assume
there exists for all any every mod congruent
""".split())
VAR_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_']*)\b")

def _variables(text: str) -> List[str]:
    """Collect plausible variable tokens, filtering out keywords and long words."""
    cand: set[str] = set()
    for m in VAR_RE.finditer(text):
        tok = m.group(1)
        low = tok.lower()
        if low in KEYWORDS:
            continue
        if len(tok) > 24:
            continue
        cand.add(tok)
    # Keep single letters or mild decorations x1, a_n, x'
    keep = [t for t in cand if re.fullmatch(r"[A-Za-z]([0-9_']+)?", t) or len(t) == 1]
    return sorted(set(keep))

def _norm_domain_token(tok: str) -> str:
    """Normalize a domain token to a canonical tag when known; otherwise leave it."""
    t = tok.strip().lower().replace(" ", "")
    return DOMAIN_MAP.get(t, t)

def _assign_domains_from_phrase(text: str, variables: List[str], doms: Dict[str,str]) -> None:
    """Assign domains from 'x is integer' or 'x,y are reals' phrases."""
    for pat,_ in PHRASE_PATTERNS:
        for m in pat.finditer(text):
            var_grp = m.group(1)
            dom_word = m.group("dom").lower()
            can_dom = PHRASE_DOMAIN.get(dom_word, dom_word)
            for v in [x.strip() for x in var_grp.split(",")]:
                if v in variables:
                    doms[v] = can_dom

def _assign_domains_from_in(text: str, variables: List[str], doms: Dict[str,str]) -> None:
    """Assign domains from membership forms: x ∈ Z, x in R, z ∈ C, etc."""
    t = text.replace("∈", " ∈ ")
    for pat in IN_PATTERNS:
        for m in pat.finditer(t):
            var_spec = m.group(1)
            raw_dom = m.group(2)
            can_dom = _norm_domain_token(raw_dom)
            for v in [x.strip() for x in var_spec.split(",")]:
                if v in variables:
                    doms[v] = can_dom

def _extract_domains(text: str, variables: List[str]) -> Dict[str,str]:
    """Combine phrase and membership-based domain assignment."""
    doms: Dict[str,str] = {}
    _assign_domains_from_phrase(text, variables, doms)
    _assign_domains_from_in(text, variables, doms)
    return doms

# --- public API ---------------------------------------------------------------

def normalize_user_input(raw: str) -> "Normalized":
    """
    Main entrypoint for normalization:
    1) strip simple LaTeX wrappers
    2) normalize symbols and spacing
    3) conservative fuzzy token fixes
    4) extract structure and return Normalized
    """
    if not raw:
        return Normalized("", [], [], [], [], [], [], [], None, None, None, [], {})

    s = _strip_latex(raw.strip())
    for pat, repl in CORRECTIONS:
        s = pat.sub(repl, s)
    s = _normalize_symbols(s)
    s = _clean_ws(s)

    # Light fuzzy fixes for single-token typos
    toks = [_fuzzy_fix_token(t) for t in s.split()]
    s = " ".join(toks)

    goal = _infer_goal(s)
    f_sig = _function_signature(s)
    sents = sentence_split(s)

    eqs   = _extract_equations(s)
    ineqs = _extract_inequalities(s)
    congs = _extract_congruences(s)
    divis = _extract_divisibility(s)
    ivals = _extract_intervals(s)
    structs = _detect_structures(s)

    vars_ = _variables(s)
    doms  = _extract_domains(s, vars_)

    quant = "universal" if FORALL_RE.search(s) else ("existential" if EXISTS_RE.search(s) else None)

    return Normalized(
        clean_text=s,
        sentences=_dedupe(sents),
        equations=_dedupe(eqs),
        inequalities=_dedupe(ineqs),
        congruences=_dedupe(congs),
        divisibility=_dedupe(divis),
        intervals=_dedupe(ivals),
        structures=structs,
        goal=goal,
        quantifiers=quant,
        function_sig=f_sig,
        variables=vars_,
        domains=doms,
    )
