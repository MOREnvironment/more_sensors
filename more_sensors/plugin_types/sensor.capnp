@0x9a783bf27d4a62e1;

using Anot = import "rpp_common/anot.capnp";

# Shared category and graph contract for every sensor plugin.
interface Sensor $Anot.plugin("Sensor") {
    graph @0 () -> (graph :SensorPayload);
}

struct IODescription {
    size @0 :UInt32;
    min @1 :List(Float64);
    max @2 :List(Float64);
    name @3 :Text;
    description @4 :Text;
}

struct StateDescription {
    size @0 :UInt32;
    min @1 :List(Float64);
    max @2 :List(Float64);
    ic @3 :List(Float64);
    name @4 :Text;
    description @5 :Text;
}

# Shared graph serialization used by every sensor plugin.
struct SensorPayload {
    inputDescription @0 :List(IODescription);
    outputDescription @1 :List(IODescription);
    stateDescription @2 :List(StateDescription);
    # Empty when the sensor graph has no state dynamics.
    dynamics @3 :Data;
    output @4 :Data;
    # Message identifier interpreted by consumers.
    messageName @5 :Text;
    # Stochastic errors applied by the consumer to the graph output.
    noise @6 :NoiseDescription;
    # Measurement rate in Hz. Zero samples on every consumer step.
    rateHz @7 :Float64;
    mounting @8 :Mounting;
    # Topic the consumer publishes on. Empty uses the consumer default.
    topic @9 :Text;
    # Messages built from the noisy output. Empty means one message named
    # messageName on topic that covers the whole output.
    messages @10 :List(MessageDescription);
}

# One message derived from consecutive elements of the noisy sensor output:
#   values = scale * output[outputIndex : outputIndex + size] + offset
# Several messages may read the same elements, so one measurement can be
# reported in more than one unit.
struct MessageDescription {
    name @0 :Text;
    # Empty uses the consumer default for this message.
    topic @1 :Text;
    outputIndex @2 :UInt32;
    size @3 :UInt32;
    # Empty lists leave the values unscaled and without an offset.
    scale @4 :List(Float64);
    offset @5 :List(Float64);
    # Variances reported with the message, for messages whose covariance is
    # not in the units of their values. Empty derives them from the noise.
    variance @6 :List(Float64);
}

# Where the sensor sits on the vessel. The graph output already refers to
# this point; consumers use it to label the measurement frame.
struct Mounting {
    # Frame of the measurement. Empty uses the consumer's vessel frame.
    frameId @0 :Text;
    # Position of that frame in the vessel body frame, in metres.
    location @1 :List(Float64);
    # Whether the consumer broadcasts the vessel-to-sensor transform.
    publishTf @2 :Bool;
}

# Per-element measurement errors in output units. An empty list disables that
# term. The graph output stays the ideal measurement, so consumers apply:
#   measurement = scaleFactor * output + bias_k + whiteNoiseStd * N(0, 1)
#   bias_{k+1} = bias_k + biasRandomWalkStd * sqrt(dt) * N(0, 1)
struct NoiseDescription {
    enabled @0 :Bool;
    seed @1 :UInt64;
    scaleFactor @2 :List(Float64);
    bias @3 :List(Float64);
    # Standard deviation per sample.
    whiteNoiseStd @4 :List(Float64);
    # Standard deviation per square root of a second.
    biasRandomWalkStd @5 :List(Float64);
    # Probability in [0, 1] that a due measurement is not produced.
    dropoutProbability @6 :Float64;
}
