from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_py.parameter_description import ParameterDescription

from more_sensors.models import sbl_sensor_casadi

from ._truth_sensor import (
    SENSOR_PARAMETERS,
    noise_parameters,
    noise_terms,
    sensor_settings,
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
        self._settings = None

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
        self._settings = sensor_settings(context, [noise_terms(context)])

    def graph(self) -> Sensor.SensorPayload:
        if self._model is None:
            raise RuntimeError("SBL must be initialized before graph()")
        return truth_sensor_payload(
            self._model,
            [("position", 3)],
            "PointStamped",
            settings=self._settings,
        )
