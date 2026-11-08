"""Frozen scalar-horizon24 model adapters and explicit matched controls."""
from .adapters import ScalarAdapter, ScaleBridge, build_model
from .controls import build_control_model, build_capacity_model
from .registry import MODEL_NAMES
from .vendor.phystok.strong_vtt_model import StrongVTT as PhysTok, StrongVTTSwitches

__all__ = ["PhysTok", "StrongVTTSwitches", "ScalarAdapter", "ScaleBridge", "build_model", "build_control_model", "build_capacity_model", "MODEL_NAMES"]
