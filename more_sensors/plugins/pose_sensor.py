import casadi as ca

from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_schema.more_sensors.IODescription import IODescription

from more_common.casadi_graph import graph_to_bytes
from more_sensors.models import pose_sensor_casadi


class PoseSensor(Sensor):
    def __init__(self) -> None:
        self._model = None

    def initialize(self, context: ComponentContext) -> None:
        self._model = pose_sensor_casadi()

    def graph(self) -> Sensor.CasadyPayload:
        if self._model is None:
            raise RuntimeError("PoseSensor must be initialized before graph()")

        payload = Sensor.CasadyPayload()
        payload.inputDescription.append(IODescription(12, name="vessel_state"))
        payload.outputDescription.append(IODescription(6, name="pose"))

        state = ca.SX.sym("state", 0)
        vessel_state = ca.SX.sym("input", 12)
        payload.output = graph_to_bytes(
            ca.Function(
                "output",
                [state, vessel_state],
                [self._model(vessel_state)],
            )
        )
        return payload
