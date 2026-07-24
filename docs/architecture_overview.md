# Aegis-V2X — Architecture Overview (Phase 1)

Source of truth: `01_Aegis-V2X_Project_Proposal`, `02_Aegis-V2X_System_Architecture`,
`03_Mathematical_Formulation`, `04_Novel_Algorithm_Design`.

## Design Principle

> Every communication decision should depend on how much the system
> currently trusts its Digital Twin.

## Seven Runtime Layers

1. **Physical Environment** — CARLA (vehicle dynamics, road geometry, weather, sensors)
2. **Wireless Environment** — NVIDIA Sionna RT (CSI, SNR, path loss, blockage, beam info)
3. **Synchronized Multimodal Inputs** — LiDAR, GPS, IMU, CSI, SNR, map/traffic context
4. **Perception & Channel Prediction** — PointPillars → V2X-ViT → GRU
5. **Digital Twin State Estimation** — `digital_twin.state.DigitalTwinState`
6. **Digital Twin Self-Evaluation** — Calibrated Twin Trust Estimator (T) + Criticality Estimator (C)
7. **TwinTrust-AP** — TAHS (horizon selection) + FSDP (joint action selection)

Followed by: Joint Adaptive Decision → Execution Layer → Runtime Feedback (closed loop).

See `architecture/system_architecture.mmd` for the full diagram and
`architecture/data_flow.mmd` / `architecture/control_flow.mmd` for the
data and control flow diagrams.

## Offline / Online Separation

Runtime inference (Trust, Criticality, TAHS, FSDP) is designed to be
lightweight (O(n) or O(1), see `03_Mathematical_Formulation` Section 10)
by moving expensive optimization (FSDP policy table generation, TAHS
parameter learning) into an **offline pipeline** that runs on the
development workstation and deploys only the resulting artifacts
(policy table, calibration parameters) to the runtime system and,
eventually, to Jetson Orin.

## Hardware Split

| Component | Runs on |
|---|---|
| CARLA, Sionna RT, training | Development workstation (RTX GPU) |
| Trust Estimator, Criticality Estimator, TAHS, FSDP (inference) | Jetson Orin (Phase 6) |
| Backend, dashboard, monitoring | Workstation / containerized (Phase 3) |

## Phase → Folder Mapping

See `docs/module_decomposition.md` for the authoritative mapping of every
repository folder to its owning phase and team member.
