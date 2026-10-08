from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from more_common.casadi_graph import graph_to_bytes
from more_sensors.models import SensorNoiseModel, SensorSampler, dvl_sensor_casadi


def _payload(rate_hz: float = 0.0, **noise) -> SimpleNamespace:
    """Build the payload fields a sampler reads for a DVL graph."""
    state = ca.SX.sym("state", 0)
    vessel_state = ca.SX.sym("input", 12)
    description = {
        "enabled": True,
        "seed": 7,
        "scaleFactor": [],
        "bias": [],
        "whiteNoiseStd": [],
        "biasRandomWalkStd": [],
        "dropoutProbability": 0.0,
    } | noise
    return SimpleNamespace(
        inputDescription=[SimpleNamespace(size=12)],
        outputDescription=[SimpleNamespace(size=3)],
        stateDescription=[],
        dynamics=b"",
        output=graph_to_bytes(
            ca.Function(
                "output",
                [state, vessel_state],
                [dvl_sensor_casadi()(vessel_state)],
            )
        ),
        messageName="TwistStamped",
        topic="",
        messages=[],
        rateHz=rate_hz,
        noise=SimpleNamespace(**description),
    )


def test_disabled_noise_returns_the_truth():
    model = SensorNoiseModel(
        3,
        enabled=False,
        bias=[1.0, 1.0, 1.0],
        white_noise_std=[1.0, 1.0, 1.0],
    )

    np.testing.assert_allclose(model.apply([1.0, 2.0, 3.0], 0.1), [1, 2, 3])
    np.testing.assert_allclose(model.white_noise_std, np.zeros(3))


def test_scale_factor_and_bias_are_deterministic():
    model = SensorNoiseModel(
        3,
        enabled=True,
        scale_factor=[2.0, 1.0, 0.5],
        bias=[0.1, -0.2, 0.3],
    )

    np.testing.assert_allclose(
        model.apply([1.0, 2.0, 4.0], 0.1),
        [2.1, 1.8, 2.3],
    )


def test_white_noise_matches_the_configured_standard_deviation():
    model = SensorNoiseModel(
        2,
        enabled=True,
        seed=1,
        white_noise_std=[0.5, 2.0],
    )

    samples = np.array([model.apply([0.0, 0.0], 0.1) for _ in range(20000)])

    np.testing.assert_allclose(samples.mean(axis=0), [0.0, 0.0], atol=0.05)
    np.testing.assert_allclose(samples.std(axis=0), [0.5, 2.0], rtol=0.03)


def test_bias_random_walk_grows_with_the_square_root_of_time():
    final_biases = []
    for seed in range(2000):
        model = SensorNoiseModel(
            1,
            enabled=True,
            seed=seed,
            bias_random_walk_std=[0.2],
        )
        for _ in range(100):
            model.apply([0.0], 0.1)
        final_biases.append(model.bias[0])

    np.testing.assert_allclose(np.std(final_biases), 0.2 * np.sqrt(10.0), rtol=0.06)


def test_same_seed_reproduces_the_measurements():
    first = SensorNoiseModel(3, enabled=True, seed=5, white_noise_std=[1] * 3)
    second = SensorNoiseModel(3, enabled=True, seed=5, white_noise_std=[1] * 3)

    np.testing.assert_allclose(
        first.apply([0.0, 0.0, 0.0], 0.1),
        second.apply([0.0, 0.0, 0.0], 0.1),
    )


def test_dropout_probability_removes_that_share_of_samples():
    model = SensorNoiseModel(
        1,
        enabled=True,
        seed=3,
        dropout_probability=0.25,
    )

    dropped = sum(model.apply([0.0], 0.1) is None for _ in range(20000))

    assert dropped / 20000 == pytest.approx(0.25, abs=0.02)


def test_noise_terms_must_match_the_output_size():
    with pytest.raises(ValueError, match="bias must contain 3 values"):
        SensorNoiseModel(3, enabled=True, bias=[0.0, 0.0])


def test_noise_standard_deviations_cannot_be_negative():
    with pytest.raises(ValueError, match="cannot be negative"):
        SensorNoiseModel(1, enabled=True, white_noise_std=[-1.0])


def test_sampler_returns_the_truth_without_noise_terms():
    sampler = SensorSampler(_payload())
    vessel_state = np.arange(12, dtype=float)

    np.testing.assert_allclose(sampler.sample(vessel_state, 0.0), [6, 7, 8])


def test_sampler_applies_the_payload_noise():
    sampler = SensorSampler(_payload(bias=[1.0, 2.0, 3.0]))
    vessel_state = np.arange(12, dtype=float)

    np.testing.assert_allclose(sampler.sample(vessel_state, 0.0), [7, 9, 11])
    np.testing.assert_allclose(sampler.truth(vessel_state), [6, 7, 8])


def test_sampler_decimates_to_the_sensor_rate():
    sampler = SensorSampler(_payload(rate_hz=5.0))
    vessel_state = np.zeros(12)

    sampled = [
        sampler.sample(vessel_state, step * 0.05) is not None
        for step in range(41)
    ]

    assert sum(sampled) == 11
    assert sampled[:5] == [True, False, False, False, True]


def test_sampler_with_zero_rate_samples_every_step():
    sampler = SensorSampler(_payload())

    assert all(
        sampler.sample(np.zeros(12), step * 0.05) is not None
        for step in range(10)
    )


def test_sampler_keeps_its_mean_rate_when_steps_jitter():
    sampler = SensorSampler(_payload(rate_hz=5.0))
    generator = np.random.default_rng(0)
    times = np.arange(2001) * 0.1 + generator.uniform(-0.004, 0.004, 2001)

    sampled = sum(
        sampler.sample(np.zeros(12), instant) is not None
        for instant in np.sort(times)
    )

    assert sampled == pytest.approx(200.0 * 5.0, abs=3)


def test_sampler_slower_than_its_rate_samples_every_step():
    sampler = SensorSampler(_payload(rate_hz=50.0))

    assert all(
        sampler.sample(np.zeros(12), step * 0.1) is not None
        for step in range(20)
    )
