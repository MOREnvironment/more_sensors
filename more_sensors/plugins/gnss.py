from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_py.parameter_description import ParameterDescription

from more_sensors.models import enu_to_geodetic_scale, gnss_sensor_casadi

from ._truth_sensor import (
    SENSOR_PARAMETERS,
    NoiseTerms,
    noise_parameters,
    noise_terms,
    sensor_settings,
    truth_sensor_payload,
)


class GNSS(Sensor):
    # Position noise is given in metres as east, north, up.
    PARAMETERS = [
        *SENSOR_PARAMETERS,
        ParameterDescription("fix_location", [0.0, 0.0, 0.0]),
        *noise_parameters("position_", scale_factor=False),
    ]

    def __init__(self) -> None:
        self._model = None
        self._settings = None

    def initialize(self, context: ComponentContext) -> None:
        fix_location = context.get_parameter("fix_location", [0.0, 0.0, 0.0])
        self._model = gnss_sensor_casadi(
            fix_location,
            context.get_parameter("location", [0.0, 0.0, 0.0]),
        )
        self._settings = sensor_settings(
            context,
            [
                self._geodetic_terms(
                    noise_terms(context, "position_"),
                    fix_location,
                )
            ],
        )

    def graph(self) -> Sensor.SensorPayload:
        if self._model is None:
            raise RuntimeError("GNSS must be initialized before graph()")
        return truth_sensor_payload(
            self._model,
            [("latitude_longitude_altitude", 3)],
            "NavSatFix",
            settings=self._settings,
        )

    @staticmethod
    def _geodetic_terms(terms: NoiseTerms, fix_location) -> NoiseTerms:
        """Convert east, north, up metres to latitude, longitude, altitude."""
        latitude_scale, longitude_scale, altitude_scale = (
            enu_to_geodetic_scale(fix_location)
        )

        def convert(values: list[float]) -> list[float]:
            east, north, up = values
            return [
                north * latitude_scale,
                east * longitude_scale,
                up * altitude_scale,
            ]

        return NoiseTerms(
            scale_factor=[1.0, 1.0, 1.0],
            bias=convert(terms.bias),
            white_noise_std=convert(terms.white_noise_std),
            bias_random_walk_std=convert(terms.bias_random_walk_std),
        )
