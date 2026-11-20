# more_sensors

Truth sensor plugins for MORE vessel simulations. The shared Sensor interface
and SensorPayload are defined in more_sensors/plugin_types/sensor.capnp.

The initial implementations consume a 12-value vessel state ordered as
position, orientation, linear velocity, and angular velocity, in the
convention of more_dynamics:

| Values | Content | Frame |
|---|---|---|
| 0-2 | position: north, east, down (m) | NED |
| 3-5 | roll, pitch, yaw (rad, ZYX) | body FRD to NED |
| 6-8 | linear velocity (m/s) | body FRD |
| 9-11 | angular velocity (rad/s) | body FRD |

- `PoseSensor`: position and orientation
- `IMU`: orientation and angular velocity
- `GNSS`: latitude, longitude, altitude of the NED position, relative to `fix_location`
- `DVL`: linear velocity
