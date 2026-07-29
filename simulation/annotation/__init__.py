"""Ground-truth label generation for Aegis-V2X Phase 2.

Implements 03_Mathematical_Formulation, Sections 5-8, using the exact
weights/temperature/beta/gamma frozen in configs/model.yaml (Phase 1). TAHS
and FSDP labelers mirror the abstract contracts in ai/twintrust_ap/tahs.py
(BaseTAHS.select_horizon) and ai/twintrust_ap/fsdp.py (BaseFSDP.discretize /
lookup_action), including reusing that module's frozen TrustBin,
CriticalityBin, and CommunicationAction enums directly — Phase 2's ground
truth and Phase 5's runtime policy must speak the same vocabulary.

Design note: configs/dataset.yaml's schema_fields does NOT include a
prediction-horizon or communication-action field. TAHS/FSDP outputs are
derived on demand from (trust, criticality) rather than stored per-sample.
`decision_labeler.py` exists here for testing/validating the bootstrap
policy against the dataset's trust/criticality distribution, not for
writing extra dataset columns.
"""

from .decision_labeler import BootstrapFSDP, BootstrapTAHS
from .future_channel_labeler import FutureChannelLabeler, FutureChannelLabel
from .trust_criticality_labeler import TrustCriticalityLabeler, TrustCriticalityLabel

__all__ = [
    "TrustCriticalityLabeler", "TrustCriticalityLabel",
    "FutureChannelLabeler", "FutureChannelLabel",
    "BootstrapTAHS", "BootstrapFSDP",
]
