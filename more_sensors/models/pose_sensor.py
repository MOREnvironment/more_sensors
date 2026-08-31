import casadi as ca


def pose_sensor_casadi() -> ca.Function:
    """Build a truth-pose sensor for a 12-value vessel state."""
    vessel_state = ca.SX.sym("vessel_state", 12)
    return ca.Function(
        "pose_sensor",
        [vessel_state],
        [vessel_state[:6]],
        ["vessel_state"],
        ["pose"],
    )
