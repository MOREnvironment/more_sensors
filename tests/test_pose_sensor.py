import casadi as ca
import numpy as np

from more_sensors.models import pose_sensor_casadi


def test_pose_sensor_returns_first_six_vessel_states():
    model = pose_sensor_casadi()
    vessel_state = np.arange(12, dtype=float)

    pose = np.asarray(model(ca.DM(vessel_state))).reshape(-1)

    np.testing.assert_allclose(pose, vessel_state[:6])
