from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_py.lifecycle import CallbackResult
from rpp_py.parameter_description import ParameterDescription

from more_sensors.models import gnss_sensor_casadi

from ._truth_sensor import SENSOR_PARAMETERS, truth_sensor_payload


class GNSS(Sensor):
    PARAMETERS = [
        *SENSOR_PARAMETERS,
        ParameterDescription("fix_location", [0.0, 0.0, 0.0]),
    ]

    def __init__(self) -> None:
        self._model = None

    def on_configure(self, context: ComponentContext) -> CallbackResult:
        self._model = gnss_sensor_casadi(
            context.get_parameter("fix_location", [0.0, 0.0, 0.0])
        )

        return CallbackResult()

    def on_cleanup(self) -> CallbackResult:
        self.__init__()
        return CallbackResult()

    def graph(self) -> Sensor.SensorPayload:
        if self._model is None:
            raise RuntimeError("GNSS must be configured before graph()")
        return truth_sensor_payload(
            self._model,
            [("latitude_longitude_altitude", 3)],
            "NavSatFix",
        )
