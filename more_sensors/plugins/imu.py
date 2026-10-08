from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_py.parameter_description import ParameterDescription

from more_sensors.models import imu_sensor_casadi

from ._truth_sensor import (
    SENSOR_PARAMETERS,
    noise_parameters,
    noise_terms,
    sensor_settings,
    truth_sensor_payload,
)


class IMU(Sensor):
    PARAMETERS = [
        *SENSOR_PARAMETERS,
        ParameterDescription("gravity", 9.80665),
        *noise_parameters("orientation_", scale_factor=False),
        *noise_parameters("gyro_"),
        *noise_parameters("accel_"),
    ]

    def __init__(self) -> None:
        self._model = None
        self._settings = None

    def initialize(self, context: ComponentContext) -> None:
        self._model = imu_sensor_casadi(
            location=context.get_parameter("location", [0.0, 0.0, 0.0]),
            gravity=context.get_parameter("gravity", 9.80665),
        )
        self._settings = sensor_settings(
            context,
            [
                noise_terms(context, "orientation_"),
                noise_terms(context, "gyro_"),
                noise_terms(context, "accel_"),
            ],
        )

    def graph(self) -> Sensor.SensorPayload:
        if self._model is None:
            raise RuntimeError("IMU must be initialized before graph()")
        return truth_sensor_payload(
            self._model,
            [
                ("orientation", 3),
                ("angular_velocity", 3),
                ("linear_acceleration", 3),
            ],
            "Imu",
            settings=self._settings,
        )
