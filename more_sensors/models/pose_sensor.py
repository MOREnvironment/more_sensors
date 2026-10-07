from collections.abc import Sequence

import casadi as ca

from .frames import sensor_position


def pose_sensor_casadi(
    location: Sequence[float] = (0.0, 0.0, 0.0),
) -> ca.Function:
    """Build a truth-pose sensor for a 12-value vessel state."""
    vessel_state = ca.SX.sym("vessel_state", 12)
    pose = ca.vertcat(
        sensor_position(vessel_state, location),
        vessel_state[3:6],
    )
    return ca.Function(
        "pose_sensor",
        [vessel_state],
        [pose],
        ["vessel_state"],
        ["pose"],
    )
