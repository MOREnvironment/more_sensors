from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_py.parameter_description import ParameterDescription

from more_sensors.models import pressure_sensor_casadi

from ._truth_sensor import (
    SENSOR_PARAMETERS,
    message_description,
    noise_parameters,
    noise_terms,
    sensor_settings,
    truth_sensor_payload,
)


class Pressure(Sensor):
    # Measures absolute pressure, with noise in pascals. Like a sensor
    # driver it derives the depth from that reading.
    PARAMETERS = [
        *SENSOR_PARAMETERS,
        ParameterDescription("report_depth", True),
        ParameterDescription("report_pressure", False),
        ParameterDescription("pressure_topic", ""),
        ParameterDescription("water_density", 1025.0),
        ParameterDescription("gravity", 9.80665),
        ParameterDescription("atmospheric_pressure", 101325.0),
        *noise_parameters(size=1),
    ]

    def __init__(self) -> None:
        self._model = None
        self._settings = None
        self._messages = []

    def initialize(self, context: ComponentContext) -> None:
        water_density = context.get_parameter("water_density", 1025.0)
        gravity = context.get_parameter("gravity", 9.80665)
        atmospheric_pressure = context.get_parameter(
            "atmospheric_pressure", 101325.0
        )
        self._model = pressure_sensor_casadi(
            location=context.get_parameter("location", [0.0, 0.0, 0.0]),
            water_density=water_density,
            gravity=gravity,
            atmospheric_pressure=atmospheric_pressure,
        )
        self._settings = sensor_settings(
            context,
            [noise_terms(context, size=1)],
        )

        pascals_per_metre = water_density * gravity
        self._messages = []
        if context.get_parameter("report_depth", True):
            self._messages.append(
                message_description(
                    "PoseWithCovarianceStamped",
                    topic=self._settings.topic or "depth",
                    scale=[-1.0 / pascals_per_metre],
                    offset=[atmospheric_pressure / pascals_per_metre],
                )
            )
        if context.get_parameter("report_pressure", False):
            self._messages.append(
                message_description(
                    "FluidPressure",
                    topic=str(
                        context.get_parameter("pressure_topic", "")
                    ).strip() or "pressure",
                )
            )
        if not self._messages:
            raise ValueError(
                "Pressure must report the depth, the pressure, or both"
            )

    def graph(self) -> Sensor.SensorPayload:
        if self._model is None:
            raise RuntimeError("Pressure must be initialized before graph()")
        return truth_sensor_payload(
            self._model,
            [("pressure", 1)],
            self._messages[0].name,
            settings=self._settings,
            messages=self._messages,
        )
