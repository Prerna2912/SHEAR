from typing import Dict, Any, List

CANONICAL_CASES = {
    "naca0012": {
        "name": "NACA 0012",
        "geometry_type": "aerofoil",
        "params": {"m": 0.0, "p": 0.0, "t": 0.12, "chord": 1.0},
        "description": "Standard symmetric airfoil profile.",
    },
    "bluff_body": {
        "name": "Bluff Body",
        "geometry_type": "bluff_body",
        "params": {"width": 1.0, "height": 0.5},
        "description": "Standard Ahmed body or rectangular bluff body.",
    },
    "cylinder": {
        "name": "Cylinder",
        "geometry_type": "cylinder",
        "params": {"radius": 0.5},
        "description": "Circular cylinder for Karman vortex street analysis.",
    },
    "flat_plate": {
        "name": "Flat Plate",
        "geometry_type": "flat_plate",
        "params": {"length": 1.0, "thickness": 0.01},
        "description": "Standard zero-pressure gradient boundary layer case.",
    },
    "ship_hull": {
        "name": "Ship Hull",
        "geometry_type": "ship_hull",
        "params": {"length": 5.0, "beam": 0.8, "draft": 0.4},
        "description": "Generic Wigley hull.",
    },
}

def list_reference_cases() -> List[str]:
    return list(CANONICAL_CASES.keys())

def get_reference_case(case_id: str) -> Dict[str, Any]:
    if case_id not in CANONICAL_CASES:
        raise ValueError(f"Unknown reference case: {case_id}")
    return CANONICAL_CASES[case_id]
