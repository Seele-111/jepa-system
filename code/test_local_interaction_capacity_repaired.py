"""Focused synthetic checks for observational and statistical failure modes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import diagnose_local_interaction_capacity_repaired as d

class GeometryTests(unittest.TestCase):
    def test_crop_preserves_short_side_and_not_full_frame(self):
        g = d.crop_geometry(720, 1280, "v")
        self.assertEqual((g["resized_height"], g["resized_width"]), (437, 778))
        self.assertEqual(g["crop_top"], 26)
        self.assertEqual(g["crop_left"], 197)
        box = g["box_xyxy"]
        self.assertAlmostEqual((box[2]-box[0])*(box[3]-box[1]), 384**2/(437*778))
        self.assertLess((box[2]-box[0])*(box[3]-box[1]), .5)
    def test_i_rounding_matches_bankers_center_crop(self):
        g = d.crop_geometry(256, 259, "i")
        self.assertEqual(g["crop_left"], 18)
        self.assertEqual(g["crop_top"], 16)
    def test_portrait_swaps_geometry(self):
        a, b = d.crop_geometry(720,1280,"v"), d.crop_geometry(1280,720,"v")
        self.assertEqual(a["resized_height"], b["resized_width"])
        self.assertEqual(a["crop_top"], b["crop_left"])
    def test_area_identity(self):
        rect = d.patch_rectangles([.2,.1,.8,.9], 4)
        w = d.overlap_weights(rect, rect)
        np.testing.assert_allclose(w, np.eye(16), atol=1e-14)
        val, ok, coverage = d.project(np.arange(16), np.ones(16,bool), w)
        np.testing.assert_allclose(val, np.arange(16))
        self.assertTrue(ok.all())
    def test_missing_source_not_observed_zero(self):
        rect = d.patch_rectangles([0,0,1,1], 2)
        val, valid, coverage = d.project([100,200,300,400], np.zeros(4,bool), d.overlap_weights(rect,rect))
        self.assertFalse(valid.any())
        self.assertTrue((coverage == 0).all())
    def test_motion_tile_pixel_edges_and_border(self):
        rect, shape = d.motion_rectangles(720,1280)
        self.assertEqual(shape, [72,128])
        self.assertAlmostEqual(rect[0,0],4/128)
        self.assertAlmostEqual(rect[0,2],43/128)
        self.assertAlmostEqual(rect[1,0],43/128)
        self.assertAlmostEqual(rect[-1,2],124/128)

class SupportTests(unittest.TestCase):
    def test_common_support_rejects_unobserved_extreme(self):
        a,b = np.arange(64,dtype=float), np.arange(64,dtype=float)
        mask = np.ones(64,bool); mask[0]=False
        a[0]=1e9
        rel = d.local_relation(a,b,mask)
        self.assertAlmostEqual(rel["cos"],1)
        self.assertEqual(rel["iou"],1)
    def test_insufficient_common_support_invalid(self):
        a=np.arange(64,dtype=float);mask=np.arange(64)<31
        self.assertIsNone(d.local_relation(a,a,mask))
        self.assertIsNone(d.motion_gain(a,a,mask))
    def test_constant_maps_do_not_have_arbitrary_topk_overlap(self):
        a=np.ones(64);mask=np.ones(64,bool)
        r=d.local_relation(a,a,mask)
        self.assertEqual(r["iou"],0)
        self.assertEqual(r["cos"],0)
        self.assertEqual(d.cross_features(a,a,mask)["vi_topk_overlap"],0)
    def test_observed_zero_motion_is_distinct_from_missing(self):
        a=np.arange(64,dtype=float)
        self.assertEqual(d.motion_gain(a,np.zeros(64),np.ones(64,bool)),0)
        self.assertIsNone(d.motion_gain(a,np.zeros(64),np.zeros(64,bool)))
    def test_first_anchor_and_motion_missing_propagate_invalid(self):
        names=[f"tile_{r}{c}_{suffix}" for r in range(3) for c in range(3) for suffix in ("residual_p90","observed_fraction")]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            v=np.arange(576,dtype=float).reshape(1,24,24)
            v=np.repeat(v,2,axis=0)
            i=np.arange(196,dtype=float).reshape(1,14,14)
            i=np.repeat(i,2,axis=0)
            np.savez(root/"raw.npz",tubelet_frame_ids=[[0,1],[2,3]],keyframe_ids=[0,3],
                     vjepa_raw_heatmaps=v,ijepa_raw_heatmaps=i,vjepa_patch_valid_mask=np.ones(v.shape,bool),
                     ijepa_patch_valid_mask=np.ones(i.shape,bool),vjepa_patch_counts=np.ones(v.shape),
                     ijepa_patch_counts=np.ones(i.shape),vjepa_valid_mask=[True,True],ijepa_valid_mask=[True,True],
                     vjepa_raw_errors=[1,2],frame_ids=np.arange(4),timestamps_sec=np.arange(4)/24)
            np.savez(root/"local.npz",signals=np.zeros((4,18)),feature_valid=np.zeros((4,18),bool),
                     feature_names=names,frame_ids=np.arange(4),fps=24.)
            geometry={"v":d.crop_geometry(720,1280,"v"),"i":d.crop_geometry(720,1280,"i"),"motion_analysis_shape":[72,128]}
            r=d.extract({"raw_path":root/"raw.npz","local_path":root/"local.npz","height":720,"width":1280,
                         "geometry":geometry,"frames":4,"fps":24.},names)
            self.assertFalse(r["valid"][0,:3].any())
            self.assertTrue(r["valid"][1,:3].all())
            self.assertFalse(r["valid"][:,5:].any())

class StatisticsTests(unittest.TestCase):
    def test_weighted_auc_ties_have_half_credit(self):
        self.assertEqual(d.weighted_auc([1,1],[1,1]),.5)
        self.assertEqual(d.weighted_auc([2],[1]),1)
        self.assertEqual(d.weighted_auc([0],[1]),0)
    def test_weighted_auc_bruteforce(self):
        p,n=np.array([1,2,3.]),np.array([0,2,4.]);pw,nw=np.array([1,2,3.]),np.array([5,2,1.])
        credit=(p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])
        expected=(credit*pw[:,None]*nw[None,:]).sum()/pw.sum()/nw.sum()
        self.assertAlmostEqual(d.weighted_auc(p,n,pw,nw),expected)
    def test_events_equal_weight_not_long_event_dominance(self):
        g=d.pooled_groups([("same",np.array([0.])),("same",np.ones(20))])
        self.assertAlmostEqual(d.weighted_auc(g["same"][0],[.5],g["same"][1],[1]),.5)
    def test_duplicate_content_not_extra_independent_group(self):
        a=d.pooled_groups([("s",[1,2]),("t",[0])]);b=d.pooled_groups([("s",[1,2]),("s",[1,2]),("t",[0])])
        neg=d.pooled_groups([("n",[.5])])
        self.assertAlmostEqual(d.group_auc_summary(a,neg)["auc"],d.group_auc_summary(b,neg)["auc"])
        self.assertEqual(d.group_auc_summary(b,neg)["positive_content_groups"],2)
    def test_group_bootstrap_fixed_and_corrected_lower_more_conservative(self):
        p=d.pooled_groups([("a",[0,2]),("b",[2,3])]);n=d.pooled_groups([("c",[1]),("d",[2])])
        a,b=d.group_auc_summary(p,n),d.group_auc_summary(p,n)
        self.assertEqual(a,b)
        self.assertLessEqual(a["bonferroni_lower"],a["ci95"][0])
    def test_strict_tail_quantile_ties(self):
        q=d.weighted_quantile(np.ones(20),np.ones(20),.95)
        self.assertEqual(q,1)
        self.assertEqual(np.mean(np.ones(20)>q),0)
    @staticmethod
    def good(name,category):
        return {"feature":name,"category":category,"event_valid_coverage_content_balanced":1.,
                "event_anchor_valid_fraction_content_balanced":1.,"normal_anchor_valid_fraction_content_balanced":1.,
                "auc":.7,"bonferroni_lower":.6,"ci95":[.6,.8],"within_video_auc_content_balanced":.7,
                "within_video_available_fraction":1.,"strict_normal_q95_event_recall_content_balanced":.4}
    def test_no_direction_flip_for_gate(self):
        a,b=self.good("x","all"),self.good("x",d.LOW);b["auc"]=.3
        self.assertFalse(d.qualifies(a,b))
    def test_cannot_combine_families_across_targets(self):
        summaries=[]
        for name in d.REPRESENTATIVES.values():
            for category in ["all",d.LOW,d.GATED]:
                s=self.good(name,category)
                if (category==d.LOW and name!="v_strength_change") or (category==d.GATED and name!="vi_strength_colocation"):
                    s["auc"]=.4
                summaries.append(s)
        self.assertFalse(d.capacity_decision(summaries)["capacity_screen_supported"])
    def test_two_base_families_same_target_allow_screen_only(self):
        summaries=[self.good(n,c) for n in d.REPRESENTATIVES.values() for c in ["all",d.LOW,d.GATED]]
        self.assertTrue(d.capacity_decision(summaries)["capacity_screen_supported"])
    def test_low_anchor_coverage_blocks_event_any_coverage(self):
        a,b=self.good("x","all"),self.good("x",d.LOW);b["event_anchor_valid_fraction_content_balanced"]=.2
        self.assertFalse(d.qualifies(a,b))

if __name__ == "__main__":
    unittest.main()
