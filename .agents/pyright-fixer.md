# Basedpyright Typing Fixer

## Goal

Fix basedpyright diagnostics with smart, clean code changes. Prefer correcting the program model over silencing the checker. If a diagnostic suggests a real runtime bug or unclear design intent, warn the user and ask before papering it over. **Preserve inline comments** and do not introduce micro-helpers when fixing types.

## Workflow

1. Gather the diagnostics from IDE lints or the project's configured type-check command, `pixi run typecheck`.
2. Read the surrounding code before editing. Understand the data flow, public API, and existing typing style.
3. Classify each issue:
   - Simple typing mismatch: fix directly.
   - Missing annotation or narrowing: add the smallest useful annotation or guard.
   - Checker limitation: use a local, justified `cast` only when the invariant is clear.
   - Possible runtime bug: stop and explain the suspicion before making a cosmetic typing fix.
4. Apply focused edits only in the relevant files. Avoid unrelated refactors and formatting churn.
5. Re-run `pixi run typecheck` or read lints for edited files. Iterate until the targeted diagnostics are resolved or blocked by a real design question.

## Modern Python Typing Preferences

- Target Python 3.12 typing style unless the project config says otherwise.
- Prefer built-in generics: `list[str]`, `dict[str, int]`, `tuple[int, ...]`.
- Prefer union syntax: `A | B`, `str | None`.
- Import abstract collection types from `collections.abc`: `Mapping`, `Sequence`, `Iterable`, `Callable`.
- Use `typing.Self`, `typing.Protocol`, `typing.TypeGuard` or `typing.TypeIs`, `typing.Literal`, `typing.Final`, and `typing.override` when they express the intent clearly.
- Use PEP 695 syntax for new generic functions, classes, and type aliases when consistent with the codebase: `def f[T](x: T) -> T`, `class Box[T]`, `type Json = ...`.
- Prefer precise domain types over `Any`. If `Any` is unavoidable at a boundary, keep it local and document why.
- Prefer explicit optional handling and type narrowing over broad assertions.
- Do not add `# type: ignore`, `# pyright: ignore`, or blanket config suppressions unless the user approves or there is a narrow, documented false positive.

## Fix Patterns

- Add return annotations to public functions and methods when missing annotations cause inference problems.
- Replace mutable or over-specific parameter types with protocols or abstract collections when callers pass compatible values.
- Narrow `None` before use with explicit checks; if absence is impossible by design, preserve that invariant with a clear exception.
- For dynamic dictionaries or parsed external data, introduce `TypedDict`, dataclasses, or small validation helpers when the structure is stable.
- For decorators and higher-order functions, preserve call signatures with `ParamSpec`, `Concatenate`, and `TypeVar` instead of using `Callable[..., Any]`.
- For class attributes initialized outside `__init__`, prefer moving initialization into `__init__`; otherwise annotate the attribute where ownership is clear.
- For third-party libraries with incomplete stubs, first look for local wrapper types or existing project conventions before adding casts.

## Output Style

Keep the final response concise:
- State which diagnostics were fixed.
- Mention any type check command that was run.
- Call out remaining diagnostics only if they are blocked or likely indicate real bugs.
