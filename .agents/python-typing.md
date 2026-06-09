# Python Typing Discipline

When editing Python files:

- Preserve basedpyright correctness for changed code.
- Use Python 3.12 typing style and prefer precise types over `Any`.
- Prefer explicit narrowing and clear invariants over broad assertions or casts.
- Do not add `# type: ignore`, `# pyright: ignore`, or config suppressions unless explicitly approved.
- After substantive Python edits, check IDE diagnostics for edited files or run `pixi run typecheck`.
- If a diagnostic may indicate a real runtime bug or unclear design intent, warn the user and ask before applying a cosmetic typing fix.
- Treat third-party unknown-type noise from pandas/numpy/torch conservatively: isolate it at boundaries with small annotations, wrappers, or local casts instead of weakening project code.
