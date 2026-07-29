"""Maps configs/simulation.yaml's frozen `town_types` to concrete CARLA map names.

configs/simulation.yaml intentionally stays CARLA-version-agnostic (it lists
abstract road-type categories, not map assets), so this concrete mapping
lives in Phase 2's own code rather than in the Phase-1-owned configs/ file.
Update here (not in configs/simulation.yaml) if a different CARLA map is
preferred for a given road type.
"""

from __future__ import annotations

TOWN_TYPE_TO_MAP = {
    "urban": "Town10HD",
    "highway": "Town06",
    "junction": "Town03",
    "roundabout": "Town03",
    "straight_road": "Town04",
}


def resolve_map(town_type: str) -> str:
    if town_type not in TOWN_TYPE_TO_MAP:
        raise KeyError(
            f"town_type '{town_type}' not in configs/simulation.yaml's frozen "
            f"carla.town_types list / TOWN_TYPE_TO_MAP {list(TOWN_TYPE_TO_MAP)}"
        )
    return TOWN_TYPE_TO_MAP[town_type]
