# Contributing to Aegis-V2X

See `docs/git_workflow.md` for branch strategy and commit conventions,
and `docs/coding_standards.md` for style requirements.

Quick start:
```bash
bash scripts/setup_env.sh
source .venv/bin/activate
make test
```

Before opening a PR:
```bash
make format
make lint
make typecheck
make test
```
