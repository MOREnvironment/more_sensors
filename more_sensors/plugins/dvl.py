from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext

from more_sensors.models import dvl_sensor_casadi

from ._truth_sensor import (
    SENSOR_PARAMETERS,
    noise_description,
    noise_parameters,
    noise_terms,
    sensor_rate,
    truth_sensor_payload,
)


class DVL(Sensor):
    PARAMETERS = [
        *SENSOR_PARAMETERS,
        *noise_parameters(),
    ]

    def __init__(self) -> None:
        self._model = None
        self._noise = None
        self._rate_hz = 0.0

    def initialize(self, context: ComponentContext) -> None:
        self._model = dvl_sensor_casadi(
            context.get_parameter("location", [0.0, 0.0, 0.0])
        )
        self._noise = noise_description(context, [noise_terms(context)])
        self._rate_hz = sensor_rate(context)

    def graph(self) -> Sensor.SensorPayload:
        if self._model is None:
            raise RuntimeError("DVL must be initialized before graph()")
        return truth_sensor_payload(
            self._model,
            [("linear_velocity", 3)],
            "TwistStamped",
            noise=self._noise,
            rate_hz=self._rate_hz,
        )
