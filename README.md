# more_sensors

Truth sensor plugins for MORE vessel simulations. The shared Sensor interface
and SensorPayload are defined in more_sensors/plugin_types/sensor.capnp.

The initial implementations consume a 12-value vessel state ordered as
position, orientation, linear velocity, and angular velocity:

- `PoseSensor`: position and orientation
- `IMU`: orientation and angular velocity
- `GNSS`: position
- `DVL`: linear velocity
