from math import pi

import casadi as ca
import numpy as np
import pytest

from more_sensors.models import (
    dvl_sensor_casadi,
    enu_to_geodetic_scale,
    gnss_sensor_casadi,
    imu_sensor_casadi,
    magnetometer_sensor_casadi,
    pose_sensor_casadi,
    pressure_sensor_casadi,
    sbl_sensor_casadi,
)


def _evaluate(model: ca.Function, vessel_state=None) -> np.ndarray:
    if vessel_state is None:
        vessel_state = np.arange(12, dtype=float)
    return np.asarray(model(ca.DM(vessel_state))).reshape(-1)


def _yawed_state() -> np.ndarray:
    """A vessel at (10, 20, -5) heading along the global y axis."""
    vessel_state = np.zeros(12)
    vessel_state[0:3] = [10.0, 20.0, -5.0]
    vessel_state[5] = pi / 2.0
    vessel_state[6:9] = [1.0, 0.0, 0.0]
    vessel_state[9:12] = [0.0, 0.0, 0.5]
    return vessel_state


def _imu(vessel_state, vessel_acceleration=None, **parameters) -> np.ndarray:
    if vessel_acceleration is None:
        vessel_acceleration = np.zeros(6)
    return _evaluate(
        imu_sensor_casadi(**parameters),
        np.concatenate([vessel_state, vessel_acceleration]),
    )


def test_imu_returns_orientation_and_angular_velocity():
    np.testing.assert_allclose(
        _imu(np.arange(12, dtype=float))[:6],
        [3.0, 4.0, 5.0, 9.0, 10.0, 11.0],
    )


def test_level_imu_at_rest_measures_gravity_on_its_z_axis():
    np.testing.assert_allclose(
        _imu(np.zeros(12), gravity=9.81)[6:9],
        [0.0, 0.0, 9.81],
    )


def test_rolled_imu_measures_gravity_on_its_y_axis():
    vessel_state = np.zeros(12)
    vessel_state[3] = pi / 2.0

    np.testing.assert_allclose(
        _imu(vessel_state, gravity=9.81)[6:9],
        [0.0, 9.81, 0.0],
        atol=1e-12,
    )


def test_imu_measures_body_and_centripetal_acceleration():
    vessel_state = np.zeros(12)
    vessel_state[6] = 4.0
    vessel_state[11] = 0.5

    np.testing.assert_allclose(
        _imu(vessel_state, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], gravity=0.0)[6:9],
        [1.0, 2.0, 0.0],
        atol=1e-12,
    )


def test_imu_lever_arm_adds_rotational_acceleration():
    vessel_state = np.zeros(12)
    vessel_state[11] = 0.5

    np.testing.assert_allclose(
        _imu(
            vessel_state,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            location=(2.0, 0.0, 0.0),
            gravity=0.0,
        )[6:9],
        [-0.5, 0.4, 0.0],
        atol=1e-12,
    )


def test_magnetometer_returns_the_field_in_the_body_frame():
    model = magnetometer_sensor_casadi((0.0, 20.0e-6, -45.0e-6))

    np.testing.assert_allclose(
        _evaluate(model, _yawed_state()),
        [20.0e-6, 0.0, -45.0e-6],
        atol=1e-18,
    )


def test_gnss_returns_geodetic_position_of_local_offset():
    latitude_scale, _, _ = enu_to_geodetic_scale()

    np.testing.assert_allclose(
        _evaluate(gnss_sensor_casadi()),
        [1.0 * latitude_scale, 0.0, 2.0],
        rtol=1e-6,
        atol=1e-9,
    )


def test_gnss_scale_matches_the_model_at_the_fix_location():
    fix_location = (45.8, 15.97, 100.0)
    latitude_scale, longitude_scale, _ = enu_to_geodetic_scale(fix_location)
    vessel_state = np.zeros(12)
    vessel_state[0:3] = [30.0, -40.0, 5.0]

    fix = _evaluate(gnss_sensor_casadi(fix_location), vessel_state)

    np.testing.assert_allclose(
        fix - np.array(fix_location),
        [-40.0 * latitude_scale, 30.0 * longitude_scale, 5.0],
        rtol=1e-4,
        atol=1e-9,
    )


def test_gnss_applies_antenna_location():
    vessel_state = _yawed_state()
    moved_state = vessel_state.copy()
    moved_state[0:3] += [0.0, 2.0, 1.0]

    np.testing.assert_allclose(
        _evaluate(
            gnss_sensor_casadi(location=(2.0, 0.0, 1.0)), vessel_state
        ),
        _evaluate(gnss_sensor_casadi(), moved_state),
        atol=1e-9,
    )


def test_dvl_returns_linear_velocity():
    np.testing.assert_allclose(
        _evaluate(dvl_sensor_casadi()),
        [6.0, 7.0, 8.0],
    )


def test_dvl_adds_velocity_induced_by_its_lever_arm():
    np.testing.assert_allclose(
        _evaluate(dvl_sensor_casadi(location=(2.0, 0.0, 0.0)), _yawed_state()),
        [1.0, 1.0, 0.0],
        atol=1e-12,
    )


def test_pose_sensor_applies_its_location():
    np.testing.assert_allclose(
        _evaluate(
            pose_sensor_casadi(location=(2.0, 0.0, 1.0)), _yawed_state()
        ),
        [10.0, 22.0, -4.0, 0.0, 0.0, pi / 2.0],
        atol=1e-12,
    )


def test_pressure_is_hydrostatic_below_the_surface():
    model = pressure_sensor_casadi(
        water_density=1000.0,
        gravity=10.0,
        atmospheric_pressure=100000.0,
    )

    np.testing.assert_allclose(_evaluate(model, _yawed_state()), [150000.0])


def test_pressure_is_atmospheric_above_the_surface():
    vessel_state = np.zeros(12)
    vessel_state[2] = 3.0

    np.testing.assert_allclose(
        _evaluate(pressure_sensor_casadi(), vessel_state),
        [101325.0],
    )


def test_pressure_applies_its_location():
    model = pressure_sensor_casadi(
        location=(0.0, 0.0, -1.0),
        water_density=1000.0,
        gravity=10.0,
        atmospheric_pressure=0.0,
    )

    np.testing.assert_allclose(_evaluate(model, _yawed_state()), [60000.0])


def test_sbl_returns_position_in_the_array_frame():
    model = sbl_sensor_casadi(
        array_position=(10.0, 10.0, 0.0),
        array_orientation=(0.0, 0.0, pi / 2.0),
    )

    np.testing.assert_allclose(
        _evaluate(model, _yawed_state()),
        [10.0, 0.0, -5.0],
        atol=1e-12,
    )


def test_sensor_location_must_have_three_values():
    with pytest.raises(ValueError, match="location must contain 3 values"):
        dvl_sensor_casadi(location=(1.0, 2.0))
