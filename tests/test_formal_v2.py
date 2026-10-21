import inspect
from liquid_depth.tof_scene_meter import ToFSceneMeter,VERSION
from liquid_depth.unified_tof import UnifiedToFMeter

def test_v2_selected_and_default_repair_enabled():
    assert VERSION=='tof_formal_v14_dynamic_wet_interior_20261021'
    assert inspect.signature(ToFSceneMeter.from_empty_geometry).parameters['evidence_repair'].default is True
    assert inspect.signature(UnifiedToFMeter).parameters['compound_repair'].default is True

def test_direct_phase_adapter_has_no_geometry_repair():
    m=ToFSceneMeter(None,sensor='tof',kind='five_phase_priors',scene='homogeneous_transparent')
    assert m.repair is None
