from collections.abc import Sequence
from math import isfinite

import casadi as ca

from .frames import body_to_global_rotation, validate_vector


def imu_sensor_casadi(
    location: Sequence[float] = (0.0, 0.0, 0.0),
    gravity: float = 9.80665,
) -> ca.Function:
    """Build a truth IMU from the vessel state and its body acceleration."""
    if not isfinite(gravity) or gravity < 0.0:
        raise ValueError("gravity cannot be negative")
    lever_arm = ca.DM(validate_vector("location", location))

    vessel_state = ca.SX.sym("vessel_state", 12)
    vessel_acceleration = ca.SX.sym("vessel_acceleration", 6)
    linear_velocity = vessel_state[6:9]
    angular_velocity = vessel_state[9:12]
    linear_acceleration = vessel_acceleration[0:3]
    angular_acceleration = vessel_acceleration[3:6]

    sensor_acceleration = (
        linear_acceleration
        + ca.cross(angular_velocity, linear_velocity)
        + ca.cross(angular_acceleration, lever_arm)
        + ca.cross(angular_velocity, ca.cross(angular_velocity, lever_arm))
    )
    gravity_reaction = body_to_global_rotation(
        vessel_state[3:6]
    ).T @ ca.DM([0.0, 0.0, gravity])

    return ca.Function(
        "imu_sensor",
        [ca.vertcat(vessel_state, vessel_acceleration)],
        [
            ca.vertcat(
                vessel_state[3:6],
                angular_velocity,
                sensor_acceleration + gravity_reaction,
            )
        ],
        ["vessel_state_and_acceleration"],
        ["imu"],
    )
