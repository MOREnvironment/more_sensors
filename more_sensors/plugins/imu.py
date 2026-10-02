from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext

from more_sensors.models import imu_sensor_casadi

from ._truth_sensor import SENSOR_PARAMETERS, truth_sensor_payload


class IMU(Sensor):
    PARAMETERS = list(SENSOR_PARAMETERS)

    def __init__(self) -> None:
        self._model = None

    def initialize(self, context: ComponentContext) -> None:
        del context
        self._model = imu_sensor_casadi()

    def graph(self) -> Sensor.SensorPayload:
        if self._model is None:
            raise RuntimeError("IMU must be initialized before graph()")
        return truth_sensor_payload(
            self._model,
            [("orientation", 3), ("angular_velocity", 3)],
            "Imu",
        )
