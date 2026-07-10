"""
CF1 — Input validation for geometry parameters.

Each geometry type has physical plausibility checks.
Raises ValueError on failure with an informative message for display in the UI.
"""

from __future__ import annotations

from typing import Dict, Any


class ValidationError(ValueError):
    """Raised when geometry parameters fail physical plausibility checks."""
    pass


# ---------------------------------------------------------------------------
# Per-geometry validators
# ---------------------------------------------------------------------------

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValidationError(msg)


def validate_aerofoil(p: Dict[str, Any]) -> None:
    chord = float(p.get("chord", 0))
    span  = float(p.get("span", 0))
    alpha = float(p.get("alpha_deg", p.get("angle_of_attack", 5.0)))
    Re    = float(p.get("Re", p.get("reynolds_number", 1e6)))
    U_inf = float(p.get("U_inf", p.get("freestream_velocity", 1.0)))
    _require(0.01 <= chord <= 20.0,  f"chord={chord} m outside [0.01, 20] m")
    _require(0.1  <= span  <= 100.0, f"span={span} m outside [0.1, 100] m")
    _require(-30  <= alpha <= 30,    f"angle_of_attack={alpha}° outside [-30°, 30°]")
    _require(1e3  <= Re    <= 1e9,   f"Re={Re:.2e} outside [1e3, 1e9]")
    _require(0.1  <= U_inf <= 300.0, f"U_inf={U_inf} m/s outside [0.1, 300]")
    naca = str(p.get("naca_code", "2412"))
    _require(len(naca) == 4 and naca.isdigit(), f"naca_code='{naca}' must be a 4-digit string")
    t = int(naca[2:]) / 100.0
    _require(0.01 <= t <= 0.40, f"NACA thickness t={t} outside [1%, 40%]")


def validate_swept_wing(p: Dict[str, Any]) -> None:
    validate_aerofoil(p)
    sweep = float(p.get("sweep_angle", 0.0))
    dihed = float(p.get("dihedral", 0.0))
    _require(-10 <= sweep <= 70, f"sweep_angle={sweep}° outside [-10°, 70°]")
    _require(-15 <= dihed <= 15, f"dihedral={dihed}° outside [-15°, 15°]")


def validate_bluff_body(p: Dict[str, Any]) -> None:
    length = float(p.get("length", 0))
    height = float(p.get("height", 0))
    speed  = float(p.get("flow_speed", p.get("U_inf", 10.0)))
    _require(0.1 <= length <= 100.0, f"length={length} m outside [0.1, 100]")
    _require(0.1 <= height <= 20.0,  f"height={height} m outside [0.1, 20]")
    _require(0.1 <= speed  <= 150.0, f"flow_speed={speed} m/s outside [0.1, 150]")


def validate_cylinder(p: Dict[str, Any]) -> None:
    D = float(p.get("diameter", 0))
    L = float(p.get("length", 0))
    V = float(p.get("flow_velocity", p.get("U_inf", 1.0)))
    _require(0.001 <= D <= 10.0,  f"diameter={D} m outside [1mm, 10m]")
    _require(0.01  <= L <= 1000.0, f"length={L} m outside [0.01, 1000] m")
    _require(0.01  <= V <= 100.0,  f"flow_velocity={V} m/s outside [0.01, 100]")


def validate_turbine_blade(p: Dict[str, Any]) -> None:
    R   = float(p.get("radius", 0))
    tsr = float(p.get("tip_speed_ratio", 7.0))
    U   = float(p.get("U_inf", p.get("freestream_velocity", 10.0)))
    pit = float(p.get("pitch_angle", 5.0))
    _require(1.0   <= R   <= 150.0, f"radius={R} m outside [1, 150] m")
    _require(1.0   <= tsr <= 20.0,  f"tip_speed_ratio={tsr} outside [1, 20]")
    _require(3.0   <= U   <= 25.0,  f"wind_speed={U} m/s outside [3, 25]")
    _require(-15.0 <= pit <= 30.0,  f"pitch_angle={pit}° outside [-15°, 30°]")


def validate_flat_plate(p: Dict[str, Any]) -> None:
    L = float(p.get("length", 0))
    W = float(p.get("width", p.get("span", L)))
    a = float(p.get("alpha_deg", p.get("angle_of_attack", 5.0)))
    _require(0.01 <= L  <= 50.0, f"length={L} m outside [0.01, 50] m")
    _require(0.01 <= W  <= 50.0, f"width={W} m outside [0.01, 50] m")
    _require(-20  <= a  <= 20,   f"angle_of_attack={a}° outside [-20°, 20°]")


def validate_ship_hull(p: Dict[str, Any]) -> None:
    L  = float(p.get("length", 0))
    B  = float(p.get("beam", 0))
    T  = float(p.get("draft", 0))
    V  = float(p.get("speed", p.get("U_inf", 5.0)))
    _require(1.0  <= L <= 500.0, f"length={L} m outside [1, 500] m")
    _require(0.5  <= B <= 100.0, f"beam={B} m outside [0.5, 100] m")
    _require(0.1  <= T <= 30.0,  f"draft={T} m outside [0.1, 30] m")
    _require(0.1  <= V <= 30.0,  f"speed={V} m/s outside [0.1, 30]")
    _require(T    <= B / 2.0,    f"draft={T} > beam/2={B/2}: hull wider than tall required")


def validate_bluff_body_wake(p: Dict[str, Any]) -> None:
    validate_bluff_body(p)
    wake = float(p.get("wake_length", 1.0))
    _require(0.5 <= wake <= 200.0, f"wake_length={wake} m outside [0.5, 200] m")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_VALIDATORS = {
    "aerofoil":         validate_aerofoil,
    "swept_wing":       validate_swept_wing,
    "bluff_body":       validate_bluff_body,
    "cylinder":         validate_cylinder,
    "turbine_blade":    validate_turbine_blade,
    "flat_plate":       validate_flat_plate,
    "ship_hull":        validate_ship_hull,
    "bluff_body_wake":  validate_bluff_body_wake,
}


def validate(geometry_type: str, params: Dict[str, Any]) -> None:
    """
    Validate geometry parameters.  Raises ValidationError with a human-readable
    message if any parameter is physically implausible.
    """
    fn = _VALIDATORS.get(geometry_type)
    if fn is None:
        raise ValidationError(
            f"Unknown geometry type '{geometry_type}'. "
            f"Supported: {list(_VALIDATORS)}"
        )
    fn(params)
