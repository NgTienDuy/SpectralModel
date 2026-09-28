"""The ``Method`` protocol, its registry, assumptions, applicability checks and shadow mode.

Brief §6.1, and the spine of milestone M2: everything that turns a spectrum into a claim
implements one interface, declares machine-readable preconditions, and is registered with a
maturity level that governs whether it may be believed.

| Module | What it holds |
| --- | --- |
| `inputs` | What a method is given, and the absent-vs-empty distinction |
| `assumptions` | Machine-readable preconditions, and the four answers a check can give |
| `applicability` | The one policy turning checks into a verdict (`PROJECT_DECISIONS.md` D13) |
| `base` | The protocol, the result type, and `execute` - the only sanctioned entry point |
| `registry` | Maturity levels with the evidence they require (D14), and file discovery |
| `shadow` | Shadow mode with the ADR-0001 §S2 fix |
| `promotion` | The gate that reads evidence and deliberately promotes nothing |
| `builtin` | The repository's own methods, one file each |
"""
