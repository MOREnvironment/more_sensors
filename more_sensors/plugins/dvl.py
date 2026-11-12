from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_py.lifecycle import CallbackResult

from more_sensors.models import dvl_sensor_casadi

from ._truth_sensor import SENSOR_PARAMETERS, truth_sensor_payload


class DVL(Sensor):
    PARAMETERS = list(SENSOR_PARAMETERS)

    def __init__(self) -> None:
        self._model = None

    def on_configure(self, context: ComponentContext) -> CallbackResult:
        del context
        self._model = dvl_sensor_casadi()

        return CallbackResult()

    def on_cleanup(self) -> CallbackResult:
        self.__init__()
        return CallbackResult()

    def graph(self) -> Sensor.SensorPayload:
        if self._model is None:
            raise RuntimeError("DVL must be configured before graph()")
        return truth_sensor_payload(
            self._model,
            [("linear_velocity", 3)],
            "TwistStamped",
        )
