from collections.abc import Sequence
from dataclasses import dataclass

import casadi as ca

from more_common.casadi_graph import graph_to_bytes
from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_py.parameter_description import ParameterDescription
from rpp_schema.more_sensors.IODescription import IODescription
from rpp_schema.more_sensors.NoiseDescription import NoiseDescription


SENSOR_PARAMETERS = (
    ParameterDescription("location", [0.0, 0.0, 0.0]),
    ParameterDescription("publish_tf", False),
    ParameterDescription("rate_hz", 0.0),
    ParameterDescription("noise_enabled", False),
    ParameterDescription("random_seed", 0),
    ParameterDescription("dropout_probability", 0.0),
)


@dataclass(frozen=True)
class NoiseTerms:
    """Noise parameters of consecutive sensor output elements."""

    scale_factor: list[float]
    bias: list[float]
    white_noise_std: list[float]
    bias_random_walk_std: list[float]


def noise_parameters(
    prefix: str = "",
    size: int = 3,
    scale_factor: bool = True,
) -> list[ParameterDescription]:
    """Declare the noise parameters of one group of output elements."""
    parameters = [
        ParameterDescription(f"{prefix}bias", [0.0] * size),
        ParameterDescription(
            f"{prefix}white_noise_std_per_sample", [0.0] * size
        ),
        ParameterDescription(
            f"{prefix}bias_random_walk_std", [0.0] * size
        ),
    ]
    if scale_factor:
        parameters.insert(
            0, ParameterDescription(f"{prefix}scale_factor", [1.0] * size)
        )
    return parameters


def noise_terms(
    context: ComponentContext,
    prefix: str = "",
    size: int = 3,
) -> NoiseTerms:
    """Read the parameters declared by noise_parameters()."""

    def read(name: str, default: float) -> list[float]:
        values = context.get_parameter(f"{prefix}{name}", [default] * size)
        if isinstance(values, (int, float)):
            values = [values] * size
        if len(values) != size:
            raise ValueError(f"{prefix}{name} must contain {size} values")
        return [float(value) for value in values]

    return NoiseTerms(
        scale_factor=read("scale_factor", 1.0),
        bias=read("bias", 0.0),
        white_noise_std=read("white_noise_std_per_sample", 0.0),
        bias_random_walk_std=read("bias_random_walk_std", 0.0),
    )


def noise_description(
    context: ComponentContext,
    terms: Sequence[NoiseTerms],
) -> NoiseDescription:
    """Build the payload noise from terms listed in output order."""
    dropout_probability = float(
        context.get_parameter("dropout_probability", 0.0)
    )
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("dropout_probability must be in [0, 1]")
    seed = int(context.get_parameter("random_seed", 0))
    if seed < 0:
        raise ValueError("random_seed cannot be negative")

    description = NoiseDescription()
    description.enabled = bool(context.get_parameter("noise_enabled", False))
    description.seed = seed
    description.dropoutProbability = dropout_probability
    for term in terms:
        description.scaleFactor.extend(term.scale_factor)
        description.bias.extend(term.bias)
        description.whiteNoiseStd.extend(term.white_noise_std)
        description.biasRandomWalkStd.extend(term.bias_random_walk_std)
    if any(value < 0.0 for value in description.whiteNoiseStd):
        raise ValueError("white_noise_std_per_sample cannot be negative")
    if any(value < 0.0 for value in description.biasRandomWalkStd):
        raise ValueError("bias_random_walk_std cannot be negative")
    return description


def sensor_rate(context: ComponentContext) -> float:
    """Read the measurement rate, where zero follows the consumer step."""
    rate_hz = float(context.get_parameter("rate_hz", 0.0))
    if rate_hz < 0.0:
        raise ValueError("rate_hz cannot be negative")
    return rate_hz


def truth_sensor_payload(
    model: ca.Function,
    output_descriptions: Sequence[tuple[str, int]],
    message_name: str,
    noise: NoiseDescription | None = None,
    rate_hz: float = 0.0,
) -> Sensor.SensorPayload:
    """Build a stateless truth-sensor graph around a vessel-state model.

    A model taking 18 values also consumes the 6-value body acceleration
    after the 12-value vessel state.
    """
    if not message_name:
        raise ValueError("message_name cannot be empty")
    input_size = model.size1_in(0)
    if input_size not in (12, 18):
        raise ValueError("sensor model must accept 12 or 18 input values")

    payload = Sensor.SensorPayload()
    payload.messageName = message_name
    payload.rateHz = rate_hz
    if noise is not None:
        payload.noise = noise
    payload.inputDescription.append(IODescription(12, name="vessel_state"))
    if input_size == 18:
        payload.inputDescription.append(
            IODescription(6, name="vessel_acceleration")
        )
    for name, size in output_descriptions:
        payload.outputDescription.append(IODescription(size, name=name))

    state = ca.SX.sym("state", 0)
    sensor_input = ca.SX.sym("input", input_size)
    payload.output = graph_to_bytes(
        ca.Function(
            "output",
            [state, sensor_input],
            [model(sensor_input)],
            ["state", "input"],
            ["output"],
        )
    )
    return payload
