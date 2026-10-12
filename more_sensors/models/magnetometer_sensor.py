from collections.abc import Sequence

import casadi as ca

from .frames import body_to_global_rotation, validate_vector


def magnetometer_sensor_casadi(
    magnetic_field: Sequence[float] = (0.0, 20.0e-6, -45.0e-6),
) -> ca.Function:
    """Build a truth magnetometer from a 12-value vessel state."""
    global_field = ca.DM(validate_vector("magnetic_field", magnetic_field))

    vessel_state = ca.SX.sym("vessel_state", 12)
    return ca.Function(
        "magnetometer_sensor",
        [vessel_state],
        [body_to_global_rotation(vessel_state[3:6]).T @ global_field],
        ["vessel_state"],
        ["magnetic_field"],
    )
