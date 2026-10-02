import casadi as ca


def imu_sensor_casadi() -> ca.Function:
    """Build a truth IMU from a 12-value vessel state."""
    vessel_state = ca.SX.sym("vessel_state", 12)
    measurement = ca.vertcat(vessel_state[3:6], vessel_state[9:12])
    return ca.Function(
        "imu_sensor",
        [vessel_state],
        [measurement],
        ["vessel_state"],
        ["imu"],
    )
