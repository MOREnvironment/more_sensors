from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_py.parameter_description import ParameterDescription

from more_sensors.models import sbl_sensor_casadi

from ._truth_sensor import (
    SENSOR_PARAMETERS,
    noise_description,
    noise_parameters,
    noise_terms,
    sensor_rate,
    truth_sensor_payload,
)


class SBL(Sensor):
    # The fix and its noise are expressed in the baseline array frame.
    PARAMETERS = [
        *SENSOR_PARAMETERS,
        ParameterDescription("array_position", [0.0, 0.0, 0.0]),
        ParameterDescription("array_orientation", [0.0, 0.0, 0.0]),
        *noise_parameters(scale_factor=False),
    ]

    def __init__(self) -> None:
        self._model = None
        self._noise = None
        self._rate_hz = 0.0

    def initialize(self, context: ComponentContext) -> None:
        self._model = sbl_sensor_casadi(
            location=context.get_parameter("location", [0.0, 0.0, 0.0]),
            array_position=context.get_parameter(
                "array_position", [0.0, 0.0, 0.0]
            ),
            array_orientation=context.get_parameter(
                "array_orientation", [0.0, 0.0, 0.0]
            ),
        )
        self._noise = noise_description(context, [noise_terms(context)])
        self._rate_hz = sensor_rate(context)

    def graph(self) -> Sensor.SensorPayload:
        if self._model is None:
            raise RuntimeError("SBL must be initialized before graph()")
        return truth_sensor_payload(
            self._model,
            [("position", 3)],
            "PointStamped",
            noise=self._noise,
            rate_hz=self._rate_hz,
        )
