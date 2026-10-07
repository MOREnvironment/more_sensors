import numpy as np
import pytest

from rpp_py.data_manager import DataManager

_DATA_MANAGER = DataManager()

from more_common.casadi_graph import RppCasadiGraph  # noqa: E402
from more_sensors.models import (  # noqa: E402
    SensorSampler,
    enu_to_geodetic_scale,
)
from more_sensors.plugins.dvl import DVL  # noqa: E402
from more_sensors.plugins.gnss import GNSS  # noqa: E402
from more_sensors.plugins.imu import IMU  # noqa: E402
from more_sensors.plugins.magnetometer import (  # noqa: E402
    Magnetometer,
)
from more_sensors.plugins.pose_sensor import PoseSensor  # noqa: E402
from more_sensors.plugins.pressure import Pressure  # noqa: E402
from more_sensors.plugins.sbl import SBL  # noqa: E402
from rpp_py.context import ComponentContext  # noqa: E402
from rpp_py.parameter_handler import ParameterHandler  # noqa: E402


SENSORS = {
    PoseSensor: ("PoseStamped", 12, 6),
    IMU: ("Imu", 18, 9),
    GNSS: ("NavSatFix", 12, 3),
    DVL: ("TwistStamped", 12, 3),
    Pressure: ("FluidPressure", 12, 1),
    SBL: ("PointStamped", 12, 3),
    Magnetometer: ("MagneticField", 12, 3),
}


def _initialized(plugin_type, **parameters):
    plugin = plugin_type()
    resolved = ParameterHandler.resolve_params(plugin.PARAMETERS, {})
    unknown = set(parameters) - set(resolved.params)
    assert not unknown, f"undeclared parameters: {unknown}"
    resolved.params.update(parameters)
    ComponentContext(instance=plugin, params=resolved).initialize()
    return plugin


@pytest.mark.parametrize("plugin_type", SENSORS)
def test_sensor_graph_is_stateless_and_declares_its_message(plugin_type):
    message_name, input_size, output_size = SENSORS[plugin_type]

    payload = _initialized(plugin_type).graph()
    graph = RppCasadiGraph(payload)

    assert payload.messageName == message_name
    assert graph.num_inputs == input_size
    assert graph.num_outputs == output_size
    assert graph.num_states == 0
    assert graph.step is None


@pytest.mark.parametrize("plugin_type", SENSORS)
def test_sensor_defaults_to_a_truth_measurement(plugin_type):
    payload = _initialized(plugin_type).graph()
    sampler = SensorSampler(payload)
    vessel_state = np.linspace(-0.3, 0.3, 12)
    vessel_acceleration = np.linspace(-0.1, 0.1, 6)

    assert not payload.noise.enabled
    assert payload.rateHz == 0.0
    np.testing.assert_allclose(
        sampler.sample(vessel_state, 0.0, vessel_acceleration),
        sampler.truth(vessel_state, vessel_acceleration),
    )


def test_only_the_imu_needs_the_vessel_acceleration():
    needs_acceleration = {
        plugin_type
        for plugin_type in SENSORS
        if SensorSampler(_initialized(plugin_type).graph()).needs_acceleration
    }

    assert needs_acceleration == {IMU}


def test_imu_without_vessel_acceleration_is_rejected():
    sampler = SensorSampler(_initialized(IMU).graph())

    with pytest.raises(ValueError, match="requires the vessel acceleration"):
        sampler.sample(np.zeros(12), 0.0)


def test_dvl_noise_parameters_reach_the_payload():
    payload = _initialized(
        DVL,
        noise_enabled=True,
        random_seed=43,
        rate_hz=5.0,
        dropout_probability=0.1,
        scale_factor=[1.01, 1.0, 1.0],
        bias=[0.02, 0.0, -0.01],
        white_noise_std_per_sample=[0.01, 0.01, 0.02],
        bias_random_walk_std=[0.001, 0.001, 0.001],
    ).graph()

    assert payload.noise.enabled
    assert payload.noise.seed == 43
    assert payload.rateHz == 5.0
    assert payload.noise.dropoutProbability == 0.1
    assert payload.noise.scaleFactor == [1.01, 1.0, 1.0]
    assert payload.noise.bias == [0.02, 0.0, -0.01]
    assert payload.noise.whiteNoiseStd == [0.01, 0.01, 0.02]
    assert payload.noise.biasRandomWalkStd == [0.001, 0.001, 0.001]


def test_imu_noise_follows_the_output_order():
    payload = _initialized(
        IMU,
        noise_enabled=True,
        orientation_bias=[0.1, 0.2, 0.3],
        gyro_bias=[0.01, 0.02, 0.03],
        gyro_scale_factor=[1.0, 1.0, 1.1],
        accel_bias=[0.4, 0.5, 0.6],
    ).graph()

    assert payload.noise.bias == [
        0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.4, 0.5, 0.6
    ]
    assert payload.noise.scaleFactor == [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0
    ]


def test_gnss_noise_in_metres_is_converted_to_fix_units():
    fix_location = [45.8, 15.97, 100.0]
    latitude_scale, longitude_scale, _ = enu_to_geodetic_scale(fix_location)

    payload = _initialized(
        GNSS,
        noise_enabled=True,
        fix_location=fix_location,
        position_bias=[3.0, -4.0, 0.5],
        position_white_noise_std_per_sample=[1.0, 2.0, 5.0],
    ).graph()

    np.testing.assert_allclose(
        payload.noise.bias,
        [-4.0 * latitude_scale, 3.0 * longitude_scale, 0.5],
    )
    np.testing.assert_allclose(
        payload.noise.whiteNoiseStd,
        [2.0 * latitude_scale, 1.0 * longitude_scale, 5.0],
    )


def test_sensor_location_is_used_as_a_lever_arm():
    vessel_state = np.zeros(12)
    vessel_state[11] = 0.5

    sampler = SensorSampler(
        _initialized(DVL, location=[2.0, 0.0, 0.0]).graph()
    )

    np.testing.assert_allclose(
        sampler.truth(vessel_state), [0.0, 1.0, 0.0], atol=1e-12
    )


def test_noisy_sensor_is_reproducible_for_a_seed():
    def measurements():
        sampler = SensorSampler(
            _initialized(
                Pressure,
                noise_enabled=True,
                random_seed=11,
                white_noise_std_per_sample=[50.0],
                bias_random_walk_std=[5.0],
            ).graph()
        )
        return [sampler.sample(np.zeros(12), step * 0.1) for step in range(20)]

    first, second = measurements(), measurements()

    np.testing.assert_allclose(first, second)
    assert np.std(first) > 0.0


def test_noise_parameter_with_a_wrong_size_is_rejected():
    with pytest.raises(ValueError, match="bias must contain 3 values"):
        _initialized(DVL, bias=[0.0, 0.0])


def test_negative_rate_is_rejected():
    with pytest.raises(ValueError, match="rate_hz cannot be negative"):
        _initialized(DVL, rate_hz=-1.0)
