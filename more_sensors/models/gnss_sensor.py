from collections.abc import Sequence
from math import cos, isfinite, pi, sin, sqrt

import casadi as ca

from .frames import sensor_position


_WGS84_SEMI_MAJOR_AXIS_METERS = 6_378_137.0
_WGS84_ECCENTRICITY_SQUARED = 6.6943799901413165e-3


def _validate_fix_location(
    fix_location: Sequence[float],
) -> tuple[float, float, float]:
    """Validate the geodetic location of the local ENU origin."""
    if len(fix_location) != 3:
        raise ValueError(
            "fix_location must contain latitude, longitude, altitude"
        )

    latitude, longitude, altitude = (float(value) for value in fix_location)
    if not all(isfinite(value) for value in (latitude, longitude, altitude)):
        raise ValueError("fix_location values must be finite")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("fix_location latitude must be in [-90, 90] degrees")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("fix_location longitude must be in [-180, 180] degrees")
    return latitude, longitude, altitude


def gnss_sensor_casadi(
    fix_location: Sequence[float] = (0.0, 0.0, 0.0),
    location: Sequence[float] = (0.0, 0.0, 0.0),
) -> ca.Function:
    """Convert a local ENU antenna position to WGS-84 coordinates."""
    latitude_deg, longitude_deg, altitude = _validate_fix_location(
        fix_location
    )
    latitude = latitude_deg * pi / 180.0
    longitude = longitude_deg * pi / 180.0

    sin_latitude = ca.sin(latitude)
    cos_latitude = ca.cos(latitude)
    sin_longitude = ca.sin(longitude)
    cos_longitude = ca.cos(longitude)
    prime_vertical_radius = _WGS84_SEMI_MAJOR_AXIS_METERS / ca.sqrt(
        1.0 - _WGS84_ECCENTRICITY_SQUARED * sin_latitude**2
    )
    origin_x = (
        prime_vertical_radius + altitude
    ) * cos_latitude * cos_longitude
    origin_y = (
        prime_vertical_radius + altitude
    ) * cos_latitude * sin_longitude
    origin_z = (
        prime_vertical_radius
        * (1.0 - _WGS84_ECCENTRICITY_SQUARED)
        + altitude
    ) * sin_latitude

    vessel_state = ca.SX.sym("vessel_state", 12)
    antenna = sensor_position(vessel_state, location)
    east, north, up = antenna[0], antenna[1], antenna[2]
    ecef_x = (
        origin_x
        - sin_longitude * east
        - sin_latitude * cos_longitude * north
        + cos_latitude * cos_longitude * up
    )
    ecef_y = (
        origin_y
        + cos_longitude * east
        - sin_latitude * sin_longitude * north
        + cos_latitude * sin_longitude * up
    )
    ecef_z = origin_z + cos_latitude * north + sin_latitude * up

    semi_minor_axis = _WGS84_SEMI_MAJOR_AXIS_METERS * ca.sqrt(
        1.0 - _WGS84_ECCENTRICITY_SQUARED
    )
    second_eccentricity_squared = (
        _WGS84_SEMI_MAJOR_AXIS_METERS**2 - semi_minor_axis**2
    ) / semi_minor_axis**2
    horizontal_distance = ca.sqrt(ecef_x**2 + ecef_y**2)
    auxiliary_angle = ca.atan2(
        ecef_z * _WGS84_SEMI_MAJOR_AXIS_METERS,
        horizontal_distance * semi_minor_axis,
    )
    geodetic_latitude = ca.atan2(
        ecef_z
        + second_eccentricity_squared
        * semi_minor_axis
        * ca.sin(auxiliary_angle) ** 3,
        horizontal_distance
        - _WGS84_ECCENTRICITY_SQUARED
        * _WGS84_SEMI_MAJOR_AXIS_METERS
        * ca.cos(auxiliary_angle) ** 3,
    )
    geodetic_longitude = ca.atan2(ecef_y, ecef_x)
    geodetic_radius = _WGS84_SEMI_MAJOR_AXIS_METERS / ca.sqrt(
        1.0
        - _WGS84_ECCENTRICITY_SQUARED * ca.sin(geodetic_latitude) ** 2
    )
    geodetic_altitude = (
        horizontal_distance / ca.cos(geodetic_latitude) - geodetic_radius
    )

    return ca.Function(
        "gnss_sensor",
        [vessel_state],
        [
            ca.vertcat(
                geodetic_latitude * 180.0 / pi,
                geodetic_longitude * 180.0 / pi,
                geodetic_altitude,
            )
        ],
        ["vessel_state"],
        ["fix"],
    )


def enu_to_geodetic_scale(
    fix_location: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Return fix units per metre of local error at the origin."""
    latitude_deg, _, altitude = _validate_fix_location(fix_location)
    latitude = latitude_deg * pi / 180.0
    denominator = 1.0 - _WGS84_ECCENTRICITY_SQUARED * sin(latitude) ** 2
    prime_vertical_radius = _WGS84_SEMI_MAJOR_AXIS_METERS / sqrt(denominator)
    meridional_radius = (
        _WGS84_SEMI_MAJOR_AXIS_METERS
        * (1.0 - _WGS84_ECCENTRICITY_SQUARED)
        / denominator**1.5
    )
    parallel_radius = (prime_vertical_radius + altitude) * cos(latitude)
    if parallel_radius <= 0.0:
        raise ValueError("fix_location cannot be at a pole")
    return (
        180.0 / pi / (meridional_radius + altitude),
        180.0 / pi / parallel_radius,
        1.0,
    )
