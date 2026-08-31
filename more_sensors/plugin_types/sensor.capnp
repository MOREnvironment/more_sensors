@0x9a783bf27d4a62e1;

using Anot = import "rpp_common/anot.capnp";

interface Sensor $Anot.plugin("Sensor") {
    graph @0 () -> (graph :CasadyPayload);
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

struct CasadyPayload {
    inputDescription @0 :List(IODescription);
    outputDescription @1 :List(IODescription);
    stateDescription @2 :List(StateDescription);
    dynamics @3 :Data;
    output @4 :Data;
}
