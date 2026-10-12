from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_py.parameter_description import ParameterDescription

from more_sensors.models import pose_sensor_casadi

from ._truth_sensor import (
    SENSOR_PARAMETERS,
    noise_parameters,
    noise_terms,
    sensor_settings,
    truth_sensor_payload,
)


class PoseSensor(Sensor):
    # with_covariance selects the message type state estimators consume.
    PARAMETERS = [
        *SENSOR_PARAMETERS,
        ParameterDescription("with_covariance", False),
        *noise_parameters("position_"),
        *noise_parameters("orientation_"),
    ]

    def __init__(self) -> None:
        self._model = None
        self._settings = None
        self._with_covariance = False

    def initialize(self, context: ComponentContext) -> None:
        self._model = pose_sensor_casadi(
            context.get_parameter("location", [0.0, 0.0, 0.0])
        )
        self._settings = sensor_settings(
            context,
            [
                noise_terms(context, "position_"),
                noise_terms(context, "orientation_"),
            ],
        )

        self._with_covariance = bool(
            context.get_parameter("with_covariance", False)
        )

    def graph(self) -> Sensor.SensorPayload:
        if self._model is None:
            raise RuntimeError("PoseSensor must be initialized before graph()")
        return truth_sensor_payload(
            self._model,
            [("position", 3), ("orientation", 3)],
            "PoseWithCovarianceStamped"
            if self._with_covariance
            else "PoseStamped",
            settings=self._settings,
        )
