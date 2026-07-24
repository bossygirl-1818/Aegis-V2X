"""
Aegis-V2X AI package.

Sub-packages (each isolated per Title07_Project_Structure Section 4):
    pointpillars/     - LiDAR spatial feature extraction   (Phase 4)
    v2x_vit/           - Cooperative multimodal fusion       (Phase 4)
    gru/               - Temporal channel prediction         (Phase 4)
    trust_estimator/   - Calibrated Twin Trust Estimator     (Phase 4)
    criticality/        - Context-Aware Criticality Estimator (Phase 4)
    twintrust_ap/       - TAHS + FSDP                         (Phase 5)
    utils/              - Shared AI utilities                 (all phases)

Phase 1 status: interfaces and package scaffolding only. No model weights,
training code, or inference logic is implemented yet.
"""
