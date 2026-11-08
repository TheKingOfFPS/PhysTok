"""Only the controls actually frozen for the manuscript; no new policies."""
from .adapters import ScalarAdapter

CONTROL_NAMES = ("delayed_xview", "strict_late_fusion", "plain_state_flatten")
CAPACITY = {"phystok_small": ("phystok",256,7021701),
            "itransformer_scaled": ("itransformer",1400,27284497),
            "timexer_scaled": ("timexer",1072,27316513)}

def build_control_model(control, candidate, layout, bridge, eval_seed=20261017):
    if control == "delayed_xview":
        from .delayed_xview import control_model
    elif control == "strict_late_fusion":
        from .strict_late_fusion import control_model
    elif control == "plain_state_flatten":
        from .plain_state_flatten import control_model
    else:
        raise ValueError(f"Unknown frozen control {control!r}; expected one of {CONTROL_NAMES}")
    return control_model(candidate, layout, bridge, eval_seed)

def build_capacity_model(variant, candidate, layout, bridge, eval_seed=20261017):
    if variant not in CAPACITY:
        raise ValueError(f"Unknown capacity variant {variant!r}")
    name,width,expected = CAPACITY[variant]
    if candidate["d_model"] != width:
        raise ValueError("Capacity width differs from the frozen experiment")
    model = ScalarAdapter(name,candidate,layout,bridge,eval_seed,capacity_control=True)
    if model.parameter_count() != expected:
        raise ValueError("Capacity model parameter count differs from frozen experiment")
    return model
