"""Protocol and metric regression tests; frozen algorithms are not patched."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from special_scene_protocol_v1 import BANDS, REPLICATES, LEVELS, SCENES, SENSORS, design, settings, tolerance, validate, protocol


class ProtocolTests(unittest.TestCase):
    def test_exact_balancing(self):
        validate()
        p=protocol()
        self.assertEqual(p['total_frames'],4752)
        self.assertEqual(len(SENSORS),2)
        self.assertEqual(len(SCENES)*len(LEVELS),9)
        self.assertEqual(len(BANDS)*REPLICATES,66)

    def test_deterministic_design_and_scope(self):
        for gid in range(66):
            d=design(gid)
            self.assertEqual(d,design(gid))
            self.assertLess(d['z']+d['nominal_liquid_height_m'],10.)
            lo,hi=BANDS[d['band']]
            self.assertGreaterEqual(d['nominal_liquid_height_m'],lo)
            self.assertLessEqual(d['nominal_liquid_height_m'],hi)

    def test_severity_controls_and_no_floor_intersection(self):
        for gid in range(66):
            d=design(gid)
            light=[settings('low_light',q,d)['light_level'] for q in LEVELS]
            self.assertTrue(light[0]>light[1]>light[2])
            amp=[settings('uneven_surface',q,d)['wave_amplitude_m'] for q in LEVELS]
            self.assertTrue(amp[0]<=amp[1]<=amp[2])
            self.assertLessEqual(amp[-1],d['nominal_liquid_height_m']*.22+1e-12)

    def test_different_tolerances_not_algorithm_changes(self):
        self.assertEqual(tolerance('uneven_surface',0.),.05)
        self.assertEqual(tolerance('glare',0.),.02)
        self.assertAlmostEqual(tolerance('uneven_surface',9.),.45)
        self.assertAlmostEqual(tolerance('uneven_surface',9.,legacy=True),.18)

    def test_all_frame_denominators_and_zero_relative(self):
        from evaluate_special_scene_strata_v1 import metric
        rows=[]
        for i,(truth,pred) in enumerate([(0.,.01),(.1,.13),(.1,None)]):
            rows.append(dict(physical_id=i,case=i,band=i,scenario='low_light',
                truth={'liquid_height_m':truth},liquid_depth_m=pred,inference_failure=None))
        m=metric(rows,'liquid_depth_m')
        self.assertAlmostEqual(m['numeric_coverage_pct'],200/3)
        self.assertAlmostEqual(m['all_frame_pass_pct'],100/3)
        self.assertEqual(m['relative_numeric_frames'],1)
        self.assertAlmostEqual(m['absrel_pct'],30.)
        self.assertAlmostEqual(sum(m['absolute_bins_all_frame_pct']),200/3)

    def test_input_evidence_has_no_liquid_truth_or_estimate(self):
        import numpy as np
        from special_scene_input_evidence_v1 import assess
        bundle={'calibration':{'intrinsics':[[100,0,7.5],[0,100,7.5],[0,0,1]],
                'camera_to_world_cv':[[1,0,0,0],[0,-1,0,0],[0,0,-1,1],[0,0,0,1]],
                'radii_m':[.05,.05],'footprint_exponent':2},
                'bottom_plane_camera':{'perpendicular_distance_m':1.},'rim_z_m':.6}
        self.assertTrue(assess(np.full((16,16),.5),bundle)['eligible'])
        self.assertFalse(assess(np.zeros((16,16)),bundle)['eligible'])
        sparse=np.zeros((16,16));sparse[7:9]=.5
        self.assertFalse(assess(sparse,bundle)['eligible'])

    def test_offline_capture_audit_does_not_reward_accuracy(self):
        import numpy as np
        from special_scene_input_evidence_v1 import assess_capture_origin
        semantic=np.full((16,16),2);origin=np.zeros((16,16),np.uint8)
        # Same support with an arbitrarily biased metric depth must qualify identically.
        a=assess_capture_origin(np.full((16,16),.5),semantic,origin,contains_liquid=True)
        b=assess_capture_origin(np.full((16,16),5.),semantic,origin,contains_liquid=True)
        self.assertEqual(a,b)
        origin[:]=2
        self.assertFalse(assess_capture_origin(np.full((16,16),.5),semantic,origin,contains_liquid=True)['eligible'])


if __name__=='__main__':
    unittest.main()
