"""Spatial descriptors: masks, position sensitivity, label-free shape and isolation."""
import copy,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
from optimized_spatial_jepa_v13 import _finite_heat_stats, SPATIAL_FIELDS, spatial_from_raw,feature_view_spatial,RECIPE,fit_spatial_member,export_spatial_member,predict_spatial_member
from test_optimized_compact_v4 import record


def spatial_record(seed):
    r=record(seed);rng=np.random.default_rng(seed)
    r.update(spatial=rng.uniform(.1,1,(r['frames'],34)).astype(np.float32),spatial_support=np.ones((r['frames'],34),bool),spatial_names=[prefix+'_spatial/'+n for prefix in ['v','i'] for n in SPATIAL_FIELDS])
    return r


class SpatialFeatures(unittest.TestCase):
    def test_empty_mask_and_nonfinite_masked_values(self):
        heat=np.full((4,4),np.nan);mask=np.zeros((4,4),bool)
        np.testing.assert_array_equal(_finite_heat_stats(heat,mask),np.zeros(17))
        mask[1,2]=1;heat[1,2]=2
        x=_finite_heat_stats(heat,mask);self.assertTrue(np.isfinite(x).all());self.assertEqual(len(x),17)
        self.assertAlmostEqual(x[12],1/3,places=6);self.assertAlmostEqual(x[13],-1/3,places=6)
        heat[1,2]=np.nan
        with self.assertRaises(ValueError):_finite_heat_stats(heat,mask)

    def test_spatial_permutation_changes_location_not_histogram(self):
        a=np.zeros((5,5));b=a.copy();a[2,1]=4;b[2,3]=4;mask=np.ones_like(a,bool)
        x=_finite_heat_stats(a,mask);y=_finite_heat_stats(b,mask)
        np.testing.assert_array_equal(x[:4],y[:4]);self.assertLess(x[4],0);self.assertGreater(y[4],0)
        self.assertAlmostEqual(x[6],y[6]);self.assertEqual(x[10],y[10])

    def test_no_labels_or_metadata_and_unsupported_values(self):
        r=spatial_record(0);q=copy.deepcopy(r);q['labels'][:]=1;q.update(name='renamed',sha256='changed',generator='unused')
        a=feature_view_spatial(RECIPE,r);b=feature_view_spatial(RECIPE,q)
        for k in [0,1]:np.testing.assert_array_equal(a[k],b[k])
        r['spatial_support'][12:15,0]=False;q=copy.deepcopy(r);q['spatial'][12:15,0]=10000
        a=feature_view_spatial(RECIPE,r);b=feature_view_spatial(RECIPE,q)
        np.testing.assert_array_equal(a[0],b[0])

    def test_raw_alignment_and_valid_anchor_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'raw.npz';heat=np.ones((2,4,4),np.float32);valid=np.ones_like(heat,bool)
            fields=dict(frame_ids=np.arange(6),timestamps_sec=np.arange(6)/10.,tubelet_frame_ids=np.array([[0,1],[4,5]]),keyframe_ids=np.array([0,5]),vjepa_raw_heatmaps=heat,vjepa_patch_valid_mask=valid,vjepa_valid_mask=np.ones(2,bool),ijepa_raw_heatmaps=heat,ijepa_patch_valid_mask=valid,ijepa_valid_mask=np.ones(2,bool))
            np.savez(p,**fields);x,mask,n=spatial_from_raw(p,6,10.)
            self.assertEqual(x.shape,(6,34));self.assertEqual(mask.shape,x.shape);self.assertEqual(len(n),34);self.assertTrue(np.isfinite(x).all())
            with self.assertRaises(ValueError):spatial_from_raw(p,6,20.)
            with self.assertRaises(ValueError):spatial_from_raw(p,7,10.)


@unittest.skipUnless(importlib.util.find_spec('sklearn'),'sklearn training environment')
class SpatialModels(unittest.TestCase):
    def test_fit_isolation_and_portable_parity(self):
        rs=[spatial_record(i) for i in range(5)];rs[0]['labels'][:]=0
        with self.assertRaises(ValueError):fit_spatial_member(RECIPE,rs,[0,1,2],[2,3],7)
        fp,vp,state=fit_spatial_member(RECIPE,rs,[0,1,2],[3],7)
        self.assertIsNone(state['transform']['pca'])
        export=export_spatial_member(state);f,v=predict_spatial_member(export,rs[3]);np.testing.assert_allclose(f,fp[3],rtol=0,atol=2e-6);self.assertAlmostEqual(v,vp[3],delta=2e-6)
        rs[4]['labels'][:]=1;rs[4]['spatial']*=1000
        again,againv,new=fit_spatial_member(RECIPE,rs,[0,1,2],[3],7)
        np.testing.assert_array_equal(fp[3],again[3]);self.assertEqual(vp[3],againv[3]);self.assertEqual(state['transform'],new['transform'])

if __name__=='__main__':unittest.main()
