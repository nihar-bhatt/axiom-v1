# Normalizes raw mathematical text and extracts structure:
# - cleans LaTeX/Unicode, fixes symbols and spacing
# - detects equations (incl. chained), inequalities, congruences, divisibility, intervals
# - extracts variables, domains, quantifiers, function signatures
# - recognizes worded divisibility/congruence, function properties, index ranges
# - outputs a Normalized dataclass (JSON-friendly)

from __future__ import annotations
import re
import difflib
from dataclasses import dataclass, field
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

# Equations: lhs = rhs; we will also split chained rhs having extra '='.
EQ_RE = re.compile(r"(?P<lhs>[^=<>!≡]+)\s*(==|=)\s*(?P<rhs>[^;.,\n]+)")

# Inequalities: ignore the '>' of arrows like '->'
INEQ_RE = re.compile(r"(?P<lhs>[^=<>!]+?)\s*(?<!-)(<=|>=|<|>)(?!>)\s*(?P<rhs>[^;.,\n]+)")

# Congruences (parenthesized and worded 'mod' forms)
CONG_RE  = re.compile(
    r"(?P<lhs>[^=≡]+)\s*≡\s*(?P<rhs>[^(\n;.,]+)\s*\(mod\s*(?P<mod>[^)]+)\)", re.I
)
CONG_WORD_RE = re.compile(
    r"(?P<lhs>[^=≡]+)\s*≡\s*(?P<rhs>[^;.,\n]+?)\s+mod(?:ulo)?\s+(?P<mod>[A-Za-z0-9_]+)",
    re.I,
)

# Divisibility: symbol and worded forms
DIV_RE   = re.compile(r"\b(?P<a>[A-Za-z0-9_)\]]+)\s*\|\s*(?P<b>[A-Za-z0-9_(\[]+)\b")
DIV_WORD_RE = re.compile(r"\b(?P<a>[A-Za-z0-9_]+)\s+divides\s+(?P<b>[A-Za-z0-9_]+)\b", re.I)
DIV_PHRASE_RE = re.compile(r"\b(?P<b>[A-Za-z0-9_]+)\s+is\s+divisible\s+by\s+(?P<a>[A-Za-z0-9_]+)\b", re.I)

# Function signatures:
#   - f : R -> R
#   - f:ℝ→ℝ
SIG_RE = re.compile(r"\b([A-Za-z])\s*[:∶]\s*([A-Za-zℝℤℚℕℂ]+)\s*[-–—]?>\s*([A-Za-zℝℤℚℕℂ]+)\b")
#   - f is a function from R to R / mapping R to R / function on R to R
FUNC_FROM_TO_RE = re.compile(
    r"\b([A-Za-z])\s+(?:is\s+(?:a\s+)?)?(?:function|mapping)\s+(?:from|on)\s*([A-Za-zℝℤℚℕℂ]+)\s*(?:to|->|→)\s*([A-Za-zℝℤℚℕℂ]+)\b",
    re.I,
)

FORALL_RE  = re.compile(r"\bfor all\b|\bfor any\b|\bfor every\b|∀", re.I)
EXISTS_RE  = re.compile(r"\bthere (exists|is)\b|∃|\bfind\b|\bdetermine\b|\bcompute\b", re.I)
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

# Keep only last expression on LHS within a sentence for equations
LEFT_EXPR_RE = re.compile(r"([A-Za-z0-9_)\]](?:[^=<>!≡;.,\n])*)$")

# Index ranges: k = 1..n, k = 1,...,n, 1 ≤ k ≤ n, for k in {1,...,n}
INDEX_EQ_DOTS = re.compile(r"\b([A-Za-z])\s*=\s*([A-Za-z0-9]+)\s*(?:\.\.|…|,?\s*\.\.\.|\s*to\s*)\s*([A-Za-z0-9]+)")
INDEX_LEQ = re.compile(r"\b([A-Za-z0-9]+)\s*(?:<=|≤)\s*([A-Za-z])\s*(?:<=|≤)\s*([A-Za-z0-9]+)")
INDEX_SET = re.compile(r"\bfor\s+([A-Za-z])\s+in\s*\{\s*1\s*,?\s*(?:\.\.\.|…)\s*,?\s*([A-Za-z0-9]+)\s*\}", re.I)

# Function/Set properties: capture into 'structures' as prop:var:property
PROP_WORDS = r"continuous|differentiable|C\^?\d*|injective|surjective|bijective|monotone|increasing|decreasing|even|odd|convex|concave|bounded|periodic|linear|affine|polynomial|orthogonal|unitary|symmetric|positive\s+definite|idempotent|invertible|homomorphism|isomorphism"
PROP_RE = re.compile(r"\b([A-Za-z])\s+is\s+(?:a\s+|an\s+)?(" + PROP_WORDS + r")\b", re.I)

# Conservative typo fixes and fuzzy vocab
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
    "there","exists","for","all","any","every","such","that","where","assume","suppose","let","then",
    "largest","greatest","maximum","minimum","least","count","number","unique"
]

def _fuzzy_fix_token(tok: str) -> str:
    cand = difflib.get_close_matches(tok, VOCAB, n=1, cutoff=0.86)
    return cand[0] if cand else tok

# --- output structure ---------------------------------------------------------

@dataclass
class Normalized:
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
    # Optional extras (safe defaults so downstream code remains compatible)
    chains: List[List[str]] = field(default_factory=list)         # chained equalities as lists
    index_ranges: List[str] = field(default_factory=list)         # e.g., "k=1..n", "1<=k<=n"
    properties: Dict[str, List[str]] = field(default_factory=dict)  # var -> [props]
    targets: List[str] = field(default_factory=list)              # hinted targets (e.g., "n" in "largest n")

# --- helpers -----------------------------------------------------------------

def _strip_latex(s: str) -> str:
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
    s = s.translate(SUPERSCRIPTS)
    for k, v in REL_MAP.items():
        s = s.replace(k, v)
    s = s.replace("^", "**")
    return s

def _clean_ws(s: str) -> str:
    s = WS.sub(" ", s)
    s = re.sub(PUNCT_FIX, r"\1 ", s)
    s = MULTISPACE.sub(" ", s).strip()
    return s

def sentence_split(text: str) -> List[str]:
    parts = re.split(SENT_SPLIT, text.strip())
    return [p.strip() for p in parts if p.strip()]

def _infer_goal(text: str) -> Optional[str]:
    tl = text.lower()
    if "prove that" in tl or tl.startswith("prove ") or " prove " in tl: return "prove"
    if "show that" in tl or tl.startswith("show "): return "show"
    if re.search(r"\bfind\b|\bdetermine\b|\bcompute\b", tl): return "find"
    if "no such" in tl or "does not exist" in tl or "nonexist" in tl: return "prove nonexistence"
    return None

def _norm_domain_token(tok: str) -> str:
    t = tok.strip().lower().replace(" ", "")
    return DOMAIN_MAP.get(t, t)

def _function_signature(text: str) -> Optional[str]:
    # Search sentence-by-sentence to reduce accidental spans
    for sent in sentence_split(text):
        t = sent
        m = SIG_RE.search(t)
        if m:
            A = _norm_domain_token(m.group(2))
            B = _norm_domain_token(m.group(3))
            return f"{m.group(1)}: {A} -> {B}"
        m2 = FUNC_FROM_TO_RE.search(t)
        if m2:
            A = _norm_domain_token(m2.group(2))
            B = _norm_domain_token(m2.group(3))
            return f"{m2.group(1)}: {A} -> {B}"
    return None

def _dedupe(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

# --- extractors ---------------------------------------------------------------

def _split_chain_terms(lhs: str, rhs: str) -> List[str]:
    # Normalize separators and split chain like "b = c = d"
    rhs_parts = [p.strip().rstrip(" .;,") for p in re.split(r"\s*=\s*", rhs)]
    terms = [lhs.strip()] + [p for p in rhs_parts if p]
    # Prune obvious noise at start of lhs if it carried leading words
    terms[0] = terms[0]
    return terms

def _extract_equations(text: str) -> Tuple[List[str], List[List[str]]]:
    """
    Returns:
      pairwise_eqs: list of "u = v"
      chains:       list of [t1, t2, t3, ...] for any chained equality found
    """
    pairwise: List[str] = []
    chains: List[List[str]] = []
    for m in EQ_RE.finditer(text):
        full = text[m.start():m.end()]
        eq_pos = full.find("=")
        if eq_pos == -1:
            continue
        left_text = full[:eq_pos]
        right_text = full[eq_pos + 1:]

        # keep only the last expression on the left within this sentence
        lhs_raw = left_text.strip()
        mm = LEFT_EXPR_RE.search(left_text)
        lhs = mm.group(1).strip() if mm else lhs_raw

        # chained split
        terms = _split_chain_terms(lhs, right_text)
        if len(terms) >= 2:
            # record chain for analysis
            if len(terms) > 2:
                chains.append(terms)
            # add pairwise
            for a, b in zip(terms, terms[1:]):
                pairwise.append(f"{a} = {b}")
    return _dedupe(pairwise), chains

def _extract_inequalities(text: str) -> List[str]:
    return [m.group(0).strip() for m in INEQ_RE.finditer(text)]

def _extract_congruences(text: str) -> List[str]:
    out = [
        f"{m.group('lhs').strip()} ≡ {m.group('rhs').strip()} (mod {m.group('mod').strip()})"
        for m in CONG_RE.finditer(text)
    ]
    out += [
        f"{m.group('lhs').strip()} ≡ {m.group('rhs').strip()} (mod {m.group('mod').strip()})"
        for m in CONG_WORD_RE.finditer(text)
    ]
    return _dedupe(out)

def _extract_divisibility(text: str) -> List[str]:
    out = [f"{m.group('a')} | {m.group('b')}" for m in DIV_RE.finditer(text)]
    out += [f"{m.group('a')} | {m.group('b')}" for m in DIV_WORD_RE.finditer(text)]
    out += [f"{m.group('a')} | {m.group('b')}" for m in DIV_PHRASE_RE.finditer(text)]
    return _dedupe(out)

def _extract_intervals(text: str) -> List[str]:
    out = []
    for m in INTERVAL_RE.finditer(text):
        l, a, b, r = m.groups()
        out.append(f"{l}{a},{b}{r}")
    return out

def _detect_structures(text: str) -> List[str]:
    found = []
    for pat, tag in STRUCT_PATTERNS:
        for _ in pat.finditer(text):
            found.append(tag)
    return _dedupe(found)

# Variables: allow x, x1, a_n, x' but ignore keywords and canonical domains
KEYWORDS = set("""
prove show determine compute integer integers rational rationals real reals
natural naturals complex primes prime such that where and or iff if then let suppose assume
there exists for all any every mod congruent largest greatest maximum minimum least count number unique
""".split())
DOMAINTOKS = {"R","Z","Q","N","C","P","R+","R-","Z+","Z-","R>=0","Z>=0","C^n","R^n"}

VAR_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_']*)\b")

def _variables(text: str) -> List[str]:
    cand: set[str] = set()
    for m in VAR_RE.finditer(text):
        tok = m.group(1)
        low = tok.lower()
        if low in KEYWORDS:
            continue
        if tok in DOMAINTOKS:
            continue
        if len(tok) > 24:
            continue
        cand.add(tok)
    keep = [t for t in cand if re.fullmatch(r"[A-Za-z]([0-9_']+)?", t) or len(t) == 1]
    return sorted(set(keep))

def _assign_domains_from_phrase(text: str, variables: List[str], doms: Dict[str,str]) -> None:
    PHRASE_PATTERNS = [
        (re.compile(r"\b([A-Za-z](?:\s*,\s*[A-Za-z])*)\s+are\s+(?P<dom>integers|rationals|reals|naturals|complex|primes)\b", re.I), "are"),
        (re.compile(r"\b([A-Za-z])\s+is\s+(?P<dom>integer|rational|real|natural|complex|prime)\b", re.I), "is"),
        (re.compile(r"\b([A-Za-z])\s+(?P<dom>complex|prime|integer|rational|real|natural)\b", re.I), "bare"),
    ]
    for pat,_ in PHRASE_PATTERNS:
        for m in pat.finditer(text):
            var_grp = m.group(1)
            dom_word = m.group("dom").lower()
            can_dom = PHRASE_DOMAIN.get(dom_word, dom_word)
            for v in [x.strip() for x in var_grp.split(",")]:
                if v in variables:
                    doms[v] = can_dom

def _assign_domains_from_in(text: str, variables: List[str], doms: Dict[str,str]) -> None:
    t = text.replace("∈", " ∈ ")
    IN_PATTERNS = [
        re.compile(r"\b([A-Za-z](?:\s*,\s*[A-Za-z])*)\s*∈\s*([A-Za-zℝℤℚℕℂP\+\>\=\d^ _\[\]()/]+)", re.I),
        re.compile(r"\b([A-Za-z](?:\s*,\s*[A-Za-z])*)\s+in\s+([A-Za-zℝℤℚℕℂP\+\>\=\d^ _\[\]()/]+)", re.I),
    ]
    for pat in IN_PATTERNS:
        for m in pat.finditer(t):
            var_spec = m.group(1)
            raw_dom = m.group(2)
            can_dom = _norm_domain_token(raw_dom)
            for v in [x.strip() for x in var_spec.split(",")]:
                if v in variables:
                    doms[v] = can_dom

def _extract_domains(text: str, variables: List[str]) -> Dict[str,str]:
    doms: Dict[str,str] = {}
    _assign_domains_from_phrase(text, variables, doms)
    _assign_domains_from_in(text, variables, doms)
    return doms

def _extract_index_ranges(text: str) -> List[str]:
    out: List[str] = []
    for m in INDEX_EQ_DOTS.finditer(text):
        out.append(f"{m.group(1)}={m.group(2)}..{m.group(3)}")
    for m in INDEX_LEQ.finditer(text):
        out.append(f"{m.group(2)} in [{m.group(1)},{m.group(3)}]")
    for m in INDEX_SET.finditer(text):
        out.append(f"{m.group(1)} in {{1..{m.group(2)}}}")
    return _dedupe(out)

def _extract_properties(text: str) -> Dict[str, List[str]]:
    props: Dict[str, List[str]] = {}
    for m in PROP_RE.finditer(text):
        v = m.group(1)
        p = re.sub(r"\s+", " ", m.group(2)).lower()
        props.setdefault(v, []).append(p)
    return props

def _goal_hints_to_structures(text: str) -> List[str]:
    hints = []
    tl = text.lower()
    if re.search(r"\blargest|greatest|maximum|max\b", tl):
        # try to grab a nearby variable name
        near = re.search(r"(largest|greatest|maximum|max)\s+(?:positive\s+|nonnegative\s+)?(?:integer\s+)?([A-Za-z][A-Za-z0-9_]*)", tl)
        if near:
            hints.append(f"goalhint:max:{near.group(2)}")
        else:
            hints.append("goalhint:max")
    if re.search(r"\bsmallest|least|minimum|min\b", tl):
        near = re.search(r"(smallest|least|minimum|min)\s+(?:positive\s+|nonnegative\s+)?(?:integer\s+)?([A-Za-z][A-Za-z0-9_]*)", tl)
        if near:
            hints.append(f"goalhint:min:{near.group(2)}")
        else:
            hints.append("goalhint:min")
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", tl):
        hints.append("goalhint:count")
    if re.search(r"\bunique\b", tl):
        hints.append("goalhint:exists_unique")
    return hints

# --- public API ---------------------------------------------------------------

def normalize_user_input(raw: str) -> "Normalized":
    if not raw:
        return Normalized("", [], [], [], [], [], [], [], None, None, None, [], {})

    s = _strip_latex(raw.strip())
    for pat, repl in CORRECTIONS:
        s = pat.sub(repl, s)
    s = _normalize_symbols(s)
    s = _clean_ws(s)
    s = " ".join(_fuzzy_fix_token(t) for t in s.split())

    goal       = _infer_goal(s)
    f_sig      = _function_signature(s)
    sents      = sentence_split(s)

    eqs, chains  = _extract_equations(s)
    ineqs        = _extract_inequalities(s)
    congs        = _extract_congruences(s)
    divis        = _extract_divisibility(s)
    ivals        = _extract_intervals(s)
    structs      = _detect_structures(s)

    # properties, goal hints, and indices (kept in optional fields; also mirror as 'structures' tags)
    props   = _extract_properties(s)
    hints   = _goal_hints_to_structures(s)
    indices = _extract_index_ranges(s)

    # reflect properties into structures as prop:var:property
    for v, ps in props.items():
        for p in ps:
            structs.append(f"prop:{v}:{p}")
    structs += hints
    structs = _dedupe(structs)

    vars_ = _variables(s)
    doms  = _extract_domains(s, vars_)

    quant = "universal" if FORALL_RE.search(s) else ("existential" if EXISTS_RE.search(s) else None)

    # optional target guess from goal hint if any
    targets: List[str] = []
    for tag in hints:
        if tag.startswith("goalhint:") and ":" in tag[9:]:
            # goalhint:max:n  -> target 'n'
            parts = tag.split(":")
            if len(parts) >= 3:
                targets.append(parts[2])

    return Normalized(
        clean_text   = s,
        sentences    = _dedupe(sents),
        equations    = _dedupe(eqs),
        inequalities = _dedupe(ineqs),
        congruences  = _dedupe(congs),
        divisibility = _dedupe(divis),
        intervals    = _dedupe(ivals),
        structures   = structs,
        goal         = goal,
        quantifiers  = quant,
        function_sig = f_sig,
        variables    = vars_,
        domains      = doms,
        chains       = chains,
        index_ranges = indices,
        properties   = props,
        targets      = _dedupe(targets),
    )
