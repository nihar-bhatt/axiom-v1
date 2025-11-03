# Axiom v1 - Formal Text Conversion

Axiom v1 converts informal mathematical text into a precise, machine-readable structure.
This is the **formal text conversion layer** of the project - the foundation for later automated reasoning and theorem proving.


## 🧰 Setup

**Requirements**

Install the dependencies directly (Python ≥ 3.10 recommended):

```bash
pip install sympy regex tqdm rapidfuzz rank-bm25
```

Use the following command to convert raw text into formal statements:
```bash
python -m src.cli.inspect_axiom --problem "problem statement"
```
