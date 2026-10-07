from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_py.parameter_description import ParameterDescription

from more_sensors.models import pressure_sensor_casadi

from ._truth_sensor import (
    SENSOR_PARAMETERS,
    noise_description,
    noise_parameters,
    noise_terms,
    sensor_rate,
    truth_sensor_payload,
)


class Pressure(Sensor):
    # Pressure noise is given in pascals.
    PARAMETERS = [
        *SENSOR_PARAMETERS,
        ParameterDescription("water_density", 1025.0),
        ParameterDescription("gravity", 9.80665),
        ParameterDescription("atmospheric_pressure", 101325.0),
        *noise_parameters(size=1),
    ]

    def __init__(self) -> None:
        self._model = None
        self._noise = None
        self._rate_hz = 0.0

    def initialize(self, context: ComponentContext) -> None:
        self._model = pressure_sensor_casadi(
            location=context.get_parameter("location", [0.0, 0.0, 0.0]),
            water_density=context.get_parameter("water_density", 1025.0),
            gravity=context.get_parameter("gravity", 9.80665),
            atmospheric_pressure=context.get_parameter(
                "atmospheric_pressure", 101325.0
            ),
        )
        self._noise = noise_description(
            context,
            [noise_terms(context, size=1)],
        )
        self._rate_hz = sensor_rate(context)

    def graph(self) -> Sensor.SensorPayload:
        if self._model is None:
            raise RuntimeError("Pressure must be initialized before graph()")
        return truth_sensor_payload(
            self._model,
            [("pressure", 1)],
            "FluidPressure",
            noise=self._noise,
            rate_hz=self._rate_hz,
        )
