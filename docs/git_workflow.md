# Aegis-V2X — Git Workflow

Per `Title07_Project_Structure_and_Software_Engineering_Guide.docx`, Section 7.

## Branches
- `main` — always deployable / paper-submission-ready
- `develop` — integration branch, target for all feature PRs
- `feature/*` — one branch per module (e.g., `feature/trust-estimator`)
- `bugfix/*`
- `experiment/*` — exploratory work that may not merge

## Process
1. Branch from `develop`: `git checkout -b feature/<name> develop`
2. Implement + `make format && make lint && make test`
3. Push, open PR into `develop`
4. Code review by at least one other team member (see `CODEOWNERS`)
5. Merge to `develop`
6. Merge `develop` → `main` only after phase verification (see
   Master Project Instructions "Phase Verification")

## Commit Messages
```
feat: implement TAHS module
fix: resolve synchronization bug
docs: update architecture diagram
test: add trust estimator unit tests
```

## Phase Tagging
Tag `main` at the end of every verified phase, e.g. `phase-1-complete`,
`phase-2-complete`, to give the team (and reviewers) a clean history of
milestones.
