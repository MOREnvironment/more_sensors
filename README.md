# more_sensors

Sensor plugins for MORE vessel simulations. Each sensor returns a stateless
CasADi graph of its ideal measurement from the 12-value vessel state
(position, orientation, linear velocity, angular velocity). The interface is
defined in more_sensors/plugin_types/sensor.capnp.

| Plugin | Output | Message | Noise prefixes |
| --- | --- | --- | --- |
| `PoseSensor` | position, orientation | `PoseStamped` | `position_`, `orientation_` |
| `IMU` | orientation, angular velocity, specific force | `Imu` | `orientation_`, `gyro_`, `accel_` |
| `Magnetometer` | body-frame field (T) | `MagneticField` | none |
| `GNSS` | latitude, longitude, altitude | `NavSatFix` | `position_` (east, north, up in m) |
| `DVL` | body-frame velocity | `TwistStamped` | none |
| `Pressure` | absolute pressure (Pa) | `FluidPressure` | none |
| `SBL` | position in the array frame | `PointStamped` | none |

The vessel state stays 12 values. The IMU alone takes a second input, the
6-value body acceleration (linear, angular), so its graph has 18 inputs.

## Parameters

Every sensor has:

- `location`: mounting position in the body frame, used as the lever arm
- `rate_hz`: measurement rate, where zero samples on every step
- `noise_enabled`, `random_seed`, `dropout_probability`

Each noise prefix adds `bias`, `white_noise_std_per_sample`,
`bias_random_walk_std` and, where it applies, `scale_factor`, with one value
per output element in output units:

    measurement = scale_factor * output + bias_k + white_noise_k
    bias_{k+1} = bias_k + bias_random_walk_std * sqrt(dt) * N(0, 1)

Noise is declared in `SensorPayload.noise` and applied by the consumer with
`more_sensors.models.SensorSampler`. It is off by default.

```python
class ComponentParameters:  # a DVL in a workspace parameters.py
    location = [1.0, 0.0, -0.3]
    rate_hz = 5.0
    noise_enabled = True
    bias = [0.02, 0.0, 0.0]
    white_noise_std_per_sample = [0.01, 0.01, 0.02]
```

## Tests

Register the library, then run `python -m pytest tests`.
