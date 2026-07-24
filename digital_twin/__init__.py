"""
Aegis-V2X Digital Twin package.

Owns the runtime representation of the Digital Twin state (DT_t) and the
abstract contracts that the Trust Estimator, Criticality Estimator, and
TwinTrust-AP modules operate on.

Implementation status: INTERFACES ONLY (Phase 1).
Full StateManager implementation is scheduled for Phase 5.
"""

from digital_twin.state import DigitalTwinState

__all__ = ["DigitalTwinState"]
