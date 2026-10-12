from collections.abc import Sequence
from dataclasses import dataclass

import casadi as ca

from more_common.casadi_graph import graph_to_bytes
from rpp_plugin_types.more_sensors import Sensor
from rpp_py.context import ComponentContext
from rpp_py.parameter_description import ParameterDescription
from rpp_schema.more_sensors.IODescription import IODescription
from rpp_schema.more_sensors.MessageDescription import MessageDescription
from rpp_schema.more_sensors.Mounting import Mounting
from rpp_schema.more_sensors.NoiseDescription import NoiseDescription


SENSOR_PARAMETERS = (
    ParameterDescription("location", [0.0, 0.0, 0.0]),
    ParameterDescription("frame_id", ""),
    ParameterDescription("publish_tf", False),
    ParameterDescription("topic", ""),
    ParameterDescription("rate_hz", 0.0),
    ParameterDescription("noise_enabled", False),
    ParameterDescription("random_seed", 0),
    ParameterDescription("dropout_probability", 0.0),
)


@dataclass(frozen=True)
class SensorSettings:
    """Payload settings every sensor reads from its parameters."""

    noise: NoiseDescription
    rate_hz: float
    mounting: Mounting
    topic: str


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


def sensor_settings(
    context: ComponentContext,
    terms: Sequence[NoiseTerms],
) -> SensorSettings:
    """Read the noise, rate, and mounting shared by every sensor."""
    rate_hz = float(context.get_parameter("rate_hz", 0.0))
    if rate_hz < 0.0:
        raise ValueError("rate_hz cannot be negative")

    location = context.get_parameter("location", [0.0, 0.0, 0.0])
    if len(location) != 3:
        raise ValueError("location must contain 3 values")
    mounting = Mounting()
    mounting.frameId = str(context.get_parameter("frame_id", "")).strip()
    mounting.location.extend(float(value) for value in location)
    mounting.publishTf = bool(context.get_parameter("publish_tf", False))
    if mounting.publishTf and not mounting.frameId:
        raise ValueError("publish_tf requires a frame_id")

    return SensorSettings(
        noise=noise_description(context, terms),
        rate_hz=rate_hz,
        mounting=mounting,
        topic=str(context.get_parameter("topic", "")).strip(),
    )


def message_description(
    name: str,
    topic: str = "",
    output_index: int = 0,
    size: int = 1,
    scale: Sequence[float] = (),
    offset: Sequence[float] = (),
    variance: Sequence[float] = (),
) -> MessageDescription:
    """Describe one message derived from part of the sensor output."""
    for term_name, term in (
        ("scale", scale),
        ("offset", offset),
        ("variance", variance),
    ):
        if len(term) not in (0, size):
            raise ValueError(f"message {term_name} must contain {size} values")
    description = MessageDescription()
    description.name = name
    description.topic = topic
    description.outputIndex = output_index
    description.size = size
    description.scale.extend(float(value) for value in scale)
    description.offset.extend(float(value) for value in offset)
    description.variance.extend(float(value) for value in variance)
    return description


def truth_sensor_payload(
    model: ca.Function,
    output_descriptions: Sequence[tuple[str, int]],
    message_name: str,
    settings: SensorSettings | None = None,
    messages: Sequence[MessageDescription] = (),
) -> Sensor.SensorPayload:
    """Build a stateless truth-sensor graph around a vessel-state model."""
    if not message_name:
        raise ValueError("message_name cannot be empty")
    input_size = model.size1_in(0)
    if input_size not in (12, 18):
        raise ValueError("sensor model must accept 12 or 18 input values")

    payload = Sensor.SensorPayload()
    payload.messageName = message_name
    if settings is not None:
        payload.noise = settings.noise
        payload.rateHz = settings.rate_hz
        payload.mounting = settings.mounting
        payload.topic = settings.topic
    payload.messages.extend(messages)
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
