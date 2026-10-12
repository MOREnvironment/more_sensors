from collections.abc import Sequence

import casadi as ca

from .frames import body_to_global_rotation, sensor_position, validate_vector


def sbl_sensor_casadi(
    location: Sequence[float] = (0.0, 0.0, 0.0),
    array_position: Sequence[float] = (0.0, 0.0, 0.0),
    array_orientation: Sequence[float] = (0.0, 0.0, 0.0),
) -> ca.Function:
    """Build a truth SBL position fix from a 12-value vessel state."""
    array_origin = ca.DM(validate_vector("array_position", array_position))
    array_to_global = body_to_global_rotation(
        ca.DM(validate_vector("array_orientation", array_orientation))
    )

    vessel_state = ca.SX.sym("vessel_state", 12)
    transponder = sensor_position(vessel_state, location)
    return ca.Function(
        "sbl_sensor",
        [vessel_state],
        [array_to_global.T @ (transponder - array_origin)],
        ["vessel_state"],
        ["position"],
    )
