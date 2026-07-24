# Aegis-V2X — Module Decomposition & Ownership Map

This table is the authoritative reference for "which phase populates
which folder" and must be kept in sync with `Title07_Project_Structure`
and `08_Project_Execution_Timeline` as those documents evolve.

| Folder | Phase | Owner | Responsibility |
|---|---|---|---|
| `docs/`, `architecture/`, `configs/` | 1 | Vaishnavi | Architecture, math formulation, config schemas |
| `simulation/`, `datasets/` | 2 | Haridharani | CARLA + Sionna RT, synchronized dataset generation |
| `backend/`, `dashboard/` | 3 | Logapriya | FastAPI, PostgreSQL, REST APIs, monitoring |
| `ai/pointpillars/`, `ai/v2x_vit/`, `ai/gru/`, `ai/trust_estimator/`, `ai/criticality/` | 4 | Vaishnavi | Perception, prediction, trust & criticality estimation |
| `ai/twintrust_ap/`, `digital_twin/` | 5 | Vaishnavi | TAHS, FSDP, Digital Twin state manager, integration |
| `deployment/` | 6 | Khushi | ONNX export, TensorRT optimization, Jetson Orin deployment |
| `evaluation/`, `papers/`, `presentations/` | 7 | Entire team (lead: Vaishnavi) | Integration, baselines, ablations, IEEE paper, thesis |
| `tests/`, `scripts/`, `.github/` | All | Entire team | Testing, tooling, CI — established Phase 1, extended every phase |
| `models/`, `results/`, `logs/`, `notebooks/` | 4-7 | Varies | Generated artifacts — not authored directly, gitignored where large |

## Module Interfaces Defined in Phase 1

These abstract contracts are frozen now so that Phase 3-6 implementers
have a stable target. Changing a signature after Phase 1 requires
updating this document and flagging the change to the whole team.

| Interface | File | Consumed By |
|---|---|---|
| `DigitalTwinState` | `digital_twin/state.py` | All estimators, TwinTrust-AP |
| `BaseDigitalTwinManager` | `digital_twin/interfaces.py` | Runtime execution layer (Phase 5) |
| `BaseTrustEstimator` | `ai/trust_estimator/base.py` | TwinTrust-AP, evaluation |
| `BaseCriticalityEstimator` | `ai/criticality/base.py` | TwinTrust-AP, evaluation |
| `BaseTAHS` | `ai/twintrust_ap/tahs.py` | `BaseTwinTrustAP` |
| `BaseFSDP` | `ai/twintrust_ap/fsdp.py` | `BaseTwinTrustAP` |
| `BaseTwinTrustAP` | `ai/twintrust_ap/policy.py` | Runtime execution layer (Phase 5) |
| `BasePointPillars` | `ai/pointpillars/base.py` | `BaseV2XViT` |
| `BaseV2XViT` | `ai/v2x_vit/base.py` | `BaseGRUPredictor` |
| `BaseGRUPredictor` | `ai/gru/base.py` | `DigitalTwinState` construction |

## Dependency Graph

```
Architecture (P1) -> Simulation (P2) -> Dataset (P2) -> Backend (P3)
    -> AI Models (P4) -> TwinTrust-AP (P5) -> Deployment (P6)
    -> Integration (P7) -> Experiments (P7) -> Paper (P7)
```
