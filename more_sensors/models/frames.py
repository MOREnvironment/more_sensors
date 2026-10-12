from collections.abc import Sequence
from math import isfinite

import casadi as ca


def validate_vector(
    name: str,
    values: Sequence[float],
    size: int = 3,
) -> tuple[float, ...]:
    """Return a finite vector of the requested size."""
    if len(values) != size:
        raise ValueError(f"{name} must contain {size} values")
    vector = tuple(float(value) for value in values)
    if not all(isfinite(value) for value in vector):
        raise ValueError(f"{name} values must be finite")
    return vector


def body_to_global_rotation(orientation: ca.SX) -> ca.SX:
    """Build the roll-pitch-yaw rotation used by the vessel dynamics."""
    roll, pitch, yaw = orientation[0], orientation[1], orientation[2]
    s_roll, c_roll = ca.sin(roll), ca.cos(roll)
    s_pitch, c_pitch = ca.sin(pitch), ca.cos(pitch)
    s_yaw, c_yaw = ca.sin(yaw), ca.cos(yaw)
    return ca.vertcat(
        ca.horzcat(
            c_yaw * c_pitch,
            c_yaw * s_pitch * s_roll - s_yaw * c_roll,
            c_yaw * s_pitch * c_roll + s_yaw * s_roll,
        ),
        ca.horzcat(
            s_yaw * c_pitch,
            s_yaw * s_pitch * s_roll + c_yaw * c_roll,
            s_yaw * s_pitch * c_roll - c_yaw * s_roll,
        ),
        ca.horzcat(
            -s_pitch,
            c_pitch * s_roll,
            c_pitch * c_roll,
        ),
    )


def sensor_position(
    vessel_state: ca.SX,
    location: Sequence[float],
) -> ca.SX:
    """Return the global position of a sensor mounted at a body location."""
    lever_arm = ca.DM(validate_vector("location", location))
    return vessel_state[0:3] + body_to_global_rotation(
        vessel_state[3:6]
    ) @ lever_arm


def sensor_linear_velocity(
    vessel_state: ca.SX,
    location: Sequence[float],
) -> ca.SX:
    """Return the body-frame velocity of a sensor at a body location."""
    lever_arm = ca.DM(validate_vector("location", location))
    return vessel_state[6:9] + ca.cross(vessel_state[9:12], lever_arm)
