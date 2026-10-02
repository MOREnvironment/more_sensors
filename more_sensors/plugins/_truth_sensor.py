from collections.abc import Sequence

import casadi as ca

from more_common.casadi_graph import graph_to_bytes
from rpp_plugin_types.more_sensors import Sensor
from rpp_py.parameter_description import ParameterDescription
from rpp_schema.more_sensors.IODescription import IODescription


SENSOR_PARAMETERS = (
    ParameterDescription("location", [0.0, 0.0, 0.0]),
    ParameterDescription("publish_tf", False),
)


def truth_sensor_payload(
    model: ca.Function,
    output_descriptions: Sequence[tuple[str, int]],
    message_name: str,
) -> Sensor.SensorPayload:
    """Build a stateless truth-sensor graph around a vessel-state model."""
    if not message_name:
        raise ValueError("message_name cannot be empty")

    payload = Sensor.SensorPayload()
    payload.messageName = message_name
    payload.inputDescription.append(IODescription(12, name="vessel_state"))
    for name, size in output_descriptions:
        payload.outputDescription.append(IODescription(size, name=name))

    state = ca.SX.sym("state", 0)
    vessel_state = ca.SX.sym("input", 12)
    payload.output = graph_to_bytes(
        ca.Function(
            "output",
            [state, vessel_state],
            [model(vessel_state)],
            ["state", "input"],
            ["output"],
        )
    )
    return payload
