import casadi as ca
import numpy as np

from more_sensors.models import (
    dvl_sensor_casadi,
    gnss_sensor_casadi,
    imu_sensor_casadi,
)


def _evaluate(model: ca.Function) -> np.ndarray:
    vessel_state = np.arange(12, dtype=float)
    return np.asarray(model(ca.DM(vessel_state))).reshape(-1)


def test_imu_returns_orientation_and_angular_velocity():
    np.testing.assert_allclose(
        _evaluate(imu_sensor_casadi()),
        [3.0, 4.0, 5.0, 9.0, 10.0, 11.0],
    )


def test_gnss_returns_position():
    np.testing.assert_allclose(
        _evaluate(gnss_sensor_casadi()),
        [0.0, 1.0, 2.0],
    )


def test_dvl_returns_linear_velocity():
    np.testing.assert_allclose(
        _evaluate(dvl_sensor_casadi()),
        [6.0, 7.0, 8.0],
    )
