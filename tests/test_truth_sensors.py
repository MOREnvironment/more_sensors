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


_WGS84_A = 6_378_137.0
_WGS84_E2 = 6.6943799901413165e-3


def _small_offset_fix(origin, north, east, down):
    """Latitude, longitude, altitude of a small NED offset, from the WGS-84 radii of curvature.

    First order in the offset: an offset d differs from the exact conversion by
    about d^2 / (2 R), so the tests use tolerances well above that and far below
    a north/east or up/down swap.
    """
    latitude = np.radians(origin[0])
    denominator = 1.0 - _WGS84_E2 * np.sin(latitude) ** 2
    meridian_radius = _WGS84_A * (1.0 - _WGS84_E2) / denominator**1.5 + origin[2]
    normal_radius = _WGS84_A / np.sqrt(denominator) + origin[2]
    return [
        origin[0] + np.degrees(north / meridian_radius),
        origin[1] + np.degrees(east / (normal_radius * np.cos(latitude))),
        origin[2] - down,
    ]


def test_gnss_reads_the_state_as_north_east_down():
    # arange(12): north 0 m, east 1 m, down 2 m.
    fix = _evaluate(gnss_sensor_casadi())
    expected = _small_offset_fix((0.0, 0.0, 0.0), 0.0, 1.0, 2.0)
    # Latitude and longitude to 1e-11 degrees (about 1 um), altitude to 1 um.
    np.testing.assert_allclose(fix[:2], expected[:2], rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(fix[2], expected[2], rtol=0.0, atol=1e-6)


def test_gnss_offsets_from_the_fix_location():
    origin = (45.0, 15.0, 100.0)
    vessel_state = np.zeros(12)
    vessel_state[:3] = [100.0, -50.0, 5.0]
    fix = np.asarray(
        gnss_sensor_casadi(origin)(ca.DM(vessel_state))
    ).reshape(-1)
    expected = _small_offset_fix(origin, 100.0, -50.0, 5.0)
    # Latitude and longitude to 1e-7 degrees (about 1 cm), altitude to 1 cm.
    np.testing.assert_allclose(fix[:2], expected[:2], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(fix[2], expected[2], rtol=0.0, atol=1e-2)


def test_dvl_returns_linear_velocity():
    np.testing.assert_allclose(
        _evaluate(dvl_sensor_casadi()),
        [6.0, 7.0, 8.0],
    )
