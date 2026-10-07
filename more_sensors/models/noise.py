from collections.abc import Sequence
from typing import Any

import casadi as ca
import numpy as np

from more_common.casadi_graph import RppCasadiGraph


class SensorNoiseModel:
    """Apply scale factor, bias, bias random walk, and white noise.

    The model follows the sensor NoiseDescription contract:
        measurement = scale_factor * truth + bias_k + white_noise_k
        bias_{k+1} = bias_k + bias_random_walk_std * sqrt(dt) * N(0, 1)
    """

    def __init__(
        self,
        size: int,
        enabled: bool = False,
        seed: int = 0,
        scale_factor: Sequence[float] = (),
        bias: Sequence[float] = (),
        white_noise_std: Sequence[float] = (),
        bias_random_walk_std: Sequence[float] = (),
        dropout_probability: float = 0.0,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be greater than zero")
        if not 0.0 <= dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be in [0, 1]")

        self.size = size
        self.enabled = bool(enabled)
        self.dropout_probability = float(dropout_probability)
        self._scale_factor = self._term("scale_factor", scale_factor, 1.0)
        self._white_noise_std = self._term(
            "white_noise_std", white_noise_std, 0.0, non_negative=True
        )
        self._bias_random_walk_std = self._term(
            "bias_random_walk_std",
            bias_random_walk_std,
            0.0,
            non_negative=True,
        )
        self._bias = self._term("bias", bias, 0.0)
        self._generator = np.random.default_rng(int(seed))

    @classmethod
    def from_description(cls, size: int, description: Any) -> "SensorNoiseModel":
        """Build the model from a sensor NoiseDescription."""
        return cls(
            size,
            enabled=description.enabled,
            seed=description.seed,
            scale_factor=description.scaleFactor,
            bias=description.bias,
            white_noise_std=description.whiteNoiseStd,
            bias_random_walk_std=description.biasRandomWalkStd,
            dropout_probability=description.dropoutProbability,
        )

    @property
    def bias(self) -> np.ndarray:
        """Current bias, including the accumulated random walk."""
        return self._bias.copy()

    @property
    def white_noise_std(self) -> np.ndarray:
        """Per-sample standard deviation, zero when noise is disabled."""
        if not self.enabled:
            return np.zeros(self.size)
        return self._white_noise_std.copy()

    def apply(self, truth: np.ndarray, delta_t: float) -> np.ndarray | None:
        """Return a noisy measurement, or None when the sample drops out."""
        truth = np.asarray(truth, dtype=float).reshape(-1)
        if truth.size != self.size:
            raise ValueError(
                f"noise model expects {self.size} values, "
                f"received {truth.size}"
            )
        if not self.enabled:
            return truth.copy()

        if delta_t > 0.0:
            self._bias += (
                self._bias_random_walk_std
                * np.sqrt(delta_t)
                * self._generator.standard_normal(self.size)
            )
        if (
            self.dropout_probability > 0.0
            and self._generator.random() < self.dropout_probability
        ):
            return None
        return (
            self._scale_factor * truth
            + self._bias
            + self._white_noise_std
            * self._generator.standard_normal(self.size)
        )

    def _term(
        self,
        name: str,
        values: Sequence[float],
        default: float,
        non_negative: bool = False,
    ) -> np.ndarray:
        if len(values) == 0:
            return np.full(self.size, default)
        term = np.asarray(values, dtype=float).reshape(-1)
        if term.size != self.size:
            raise ValueError(f"{name} must contain {self.size} values")
        if not np.all(np.isfinite(term)):
            raise ValueError(f"{name} values must be finite")
        if non_negative and np.any(term < 0.0):
            raise ValueError(f"{name} values cannot be negative")
        return term.copy()


class SensorSampler:
    """Evaluate a sensor graph at its rate and apply its noise description."""

    _VESSEL_STATE_SIZE = 12
    _VESSEL_ACCELERATION_SIZE = 6

    def __init__(self, payload: Any) -> None:
        self.graph = RppCasadiGraph(payload)
        self.message_name = str(payload.messageName)
        self.rate_hz = float(payload.rateHz)
        if self.rate_hz < 0.0:
            raise ValueError("sensor rate cannot be negative")
        self.noise = SensorNoiseModel.from_description(
            self.graph.num_outputs,
            payload.noise,
        )
        self._last_sample_time: float | None = None
        self._next_sample_time: float | None = None

    @property
    def needs_acceleration(self) -> bool:
        """Whether the graph also consumes the 6-value body acceleration."""
        return self.graph.num_inputs == (
            self._VESSEL_STATE_SIZE + self._VESSEL_ACCELERATION_SIZE
        )

    def truth(
        self,
        vessel_state: np.ndarray,
        vessel_acceleration: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the ideal measurement for a vessel state."""
        graph_input = np.asarray(vessel_state, dtype=float).reshape(-1)
        if self.needs_acceleration:
            if vessel_acceleration is None:
                raise ValueError(
                    f"sensor {self.message_name} requires the vessel "
                    "acceleration"
                )
            graph_input = np.concatenate(
                [
                    graph_input,
                    np.asarray(vessel_acceleration, dtype=float).reshape(-1),
                ]
            )
        if graph_input.size != self.graph.num_inputs:
            raise ValueError(
                f"sensor {self.message_name} expects "
                f"{self.graph.num_inputs} input values, "
                f"received {graph_input.size}"
            )
        output = self.graph.output(
            ca.DM.zeros(self.graph.num_states, 1),
            ca.DM(graph_input.reshape((-1, 1))),
        )
        return np.asarray(output.full(), dtype=float).reshape(-1)

    def sample(
        self,
        vessel_state: np.ndarray,
        time: float,
        vessel_acceleration: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return a measurement, or None when none is due or it drops out."""
        if self._last_sample_time is None:
            delta_t = 0.0
        else:
            delta_t = time - self._last_sample_time
            if delta_t < 0.0:
                raise ValueError("sensor time cannot move backwards")
        if self.rate_hz > 0.0:
            # Samples are due on a fixed schedule, so consumer steps that
            # jitter around the sample period do not lower the mean rate.
            period = 1.0 / self.rate_hz
            if self._next_sample_time is None:
                self._next_sample_time = time
            if time < self._next_sample_time - 1e-9 * period:
                return None
            self._next_sample_time += period
            if self._next_sample_time <= time:
                self._next_sample_time = time + period
        self._last_sample_time = time
        return self.noise.apply(
            self.truth(vessel_state, vessel_acceleration),
            delta_t,
        )
