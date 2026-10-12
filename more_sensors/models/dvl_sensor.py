from collections.abc import Sequence

import casadi as ca

from .frames import sensor_linear_velocity


def dvl_sensor_casadi(
    location: Sequence[float] = (0.0, 0.0, 0.0),
) -> ca.Function:
    """Build a truth DVL velocity sensor from a 12-value vessel state."""
    vessel_state = ca.SX.sym("vessel_state", 12)
    return ca.Function(
        "dvl_sensor",
        [vessel_state],
        [sensor_linear_velocity(vessel_state, location)],
        ["vessel_state"],
        ["linear_velocity"],
    )
