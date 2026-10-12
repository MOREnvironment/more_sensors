from collections.abc import Sequence
from math import isfinite

import casadi as ca

from .frames import sensor_position


def pressure_sensor_casadi(
    location: Sequence[float] = (0.0, 0.0, 0.0),
    water_density: float = 1025.0,
    gravity: float = 9.80665,
    atmospheric_pressure: float = 101325.0,
) -> ca.Function:
    """Build a truth absolute-pressure sensor from a 12-value vessel state."""
    for name, value in (
        ("water_density", water_density),
        ("gravity", gravity),
    ):
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be greater than zero")
    if not isfinite(atmospheric_pressure) or atmospheric_pressure < 0.0:
        raise ValueError("atmospheric_pressure cannot be negative")

    vessel_state = ca.SX.sym("vessel_state", 12)
    depth = ca.fmax(0.0, -sensor_position(vessel_state, location)[2])
    return ca.Function(
        "pressure_sensor",
        [vessel_state],
        [atmospheric_pressure + water_density * gravity * depth],
        ["vessel_state"],
        ["pressure"],
    )
