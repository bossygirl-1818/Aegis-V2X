# Aegis-V2X — Coding Standards

Per `Title07_Project_Structure_and_Software_Engineering_Guide.docx`, Section 6.

## Python Style
- PEP 8, enforced via `flake8` + `black` (line length 100, see `pyproject.toml`)
- Type hints on all public functions and methods
- Descriptive names; functions kept under ~50 lines where practical
- No duplicated logic — shared code goes in `ai/utils/` or module-local `utils.py`

## Documentation
Every function/method docstring must include:
```
"""
<One-line purpose>

Parameters
----------
...

Returns
-------
...

Example
-------
>>> ...
"""
```

## Naming Convention
- Classes: `PascalCase` (e.g., `DigitalTwinState`)
- Functions: `snake_case()` (e.g., `calculate_trust()`)
- Variables: `snake_case` (e.g., `trust_score`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_PREDICTION_HORIZON`)

## Enforcement
Run before every commit:
```
make format   # black + isort
make lint     # flake8 + isort --check-only
make typecheck  # mypy
make test     # pytest
```
Or install the pre-commit hook: `pre-commit install` (config to be added
alongside the first real implementation PR in Phase 3/4).
