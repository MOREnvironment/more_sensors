import casadi as ca


def dvl_sensor_casadi() -> ca.Function:
    """Build a truth DVL velocity sensor from a 12-value vessel state."""
    vessel_state = ca.SX.sym("vessel_state", 12)
    return ca.Function(
        "dvl_sensor",
        [vessel_state],
        [vessel_state[6:9]],
        ["vessel_state"],
        ["linear_velocity"],
    )
