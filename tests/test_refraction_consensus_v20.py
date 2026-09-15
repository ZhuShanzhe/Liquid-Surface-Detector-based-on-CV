import pytest
from liquid_depth.refraction_consensus_v20 import choose_result

def result(t=.03,lo=.027,hi=.033,accepted=True):
 return dict(accepted=accepted,thickness_m=t if accepted else None,candidate_thickness_m=t,
   internal_depth_m=.4+t if accepted else None,top_depth_m=.4,profile_range_m=[lo,hi],
   temporal={'frames':64},status='test')

def test_union_of_plausible_sets_prevents_false_scale_certainty():
 q=[result(lo=.025,hi=.034),result(lo=.029,hi=.038),result(lo=.027,hi=.036)]
 r=choose_result(result(),result(accepted=False),q)
 assert not r['accepted'] and not r['consensus']['consistent']

def test_frozen_fallback_remains_available_when_new_evidence_is_ambiguous():
 r=choose_result(result(accepted=False),result(),[result(accepted=False)]*3)
 assert r['accepted'] and r['selection']=='v19_fallback'

def test_methods_that_disagree_are_not_averaged_into_false_precision():
 r=choose_result(result(),result(t=.04),[result()]*3)
 assert not r['accepted'] and r['status']=='method_disagreement'

def test_legacy_cannot_override_failed_temporal_acquisition():
 r=choose_result(dict(accepted=False,status='moving_surface'),result(),[result()]*3)
 assert not r['accepted'] and r['status']=='moving_surface'

def test_selection_ignores_ground_truth_fields():
 qs=[result(t=.03),result(t=.0305),result(t=.031)]
 a=choose_result(result(),result(accepted=False),qs)
 b=choose_result(dict(result(),truth_thickness_m=.9),dict(result(accepted=False),truth_thickness_m=1.),
    [dict(q,truth_thickness_m=7.) for q in qs])
 assert a['accepted']==b['accepted']
 assert a['thickness_m']==pytest.approx(b['thickness_m'])
 assert a['liquid_identity_verified'] is False
