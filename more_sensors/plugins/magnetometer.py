from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_py.parameter_description import ParameterDescription

from more_sensors.models import magnetometer_sensor_casadi

from ._truth_sensor import (
    SENSOR_PARAMETERS,
    noise_description,
    noise_parameters,
    noise_terms,
    sensor_rate,
    truth_sensor_payload,
)


class Magnetometer(Sensor):
    # The field and its noise are given in tesla.
    PARAMETERS = [
        *SENSOR_PARAMETERS,
        ParameterDescription("magnetic_field", [0.0, 20.0e-6, -45.0e-6]),
        *noise_parameters(),
    ]

    def __init__(self) -> None:
        self._model = None
        self._noise = None
        self._rate_hz = 0.0

    def initialize(self, context: ComponentContext) -> None:
        self._model = magnetometer_sensor_casadi(
            context.get_parameter("magnetic_field", [0.0, 20.0e-6, -45.0e-6])
        )
        self._noise = noise_description(context, [noise_terms(context)])
        self._rate_hz = sensor_rate(context)

    def graph(self) -> Sensor.SensorPayload:
        if self._model is None:
            raise RuntimeError(
                "Magnetometer must be initialized before graph()"
            )
        return truth_sensor_payload(
            self._model,
            [("magnetic_field", 3)],
            "MagneticField",
            noise=self._noise,
            rate_hz=self._rate_hz,
        )
