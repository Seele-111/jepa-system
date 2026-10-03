"""Regression for the independent-review support-mask defect, not score mirroring."""
import copy,importlib.util,unittest
import numpy as np
from test_spatial_jepa_v13 import spatial_record
from optimized_spatial_jepa_v13r import masked_video_statistics,feature_view_spatial_repaired,fit_repaired_video
from optimized_spatial_jepa_v13 import feature_view_spatial,RECIPE


class SpatialMaskRepair(unittest.TestCase):
    def test_unsupported_values_cannot_change_either_branch(self):
        r=spatial_record(0);r['spatial_support'][0:4,:]=False;q=copy.deepcopy(r);q['spatial'][0:4,:]+=10000
        a=feature_view_spatial_repaired(r);b=feature_view_spatial_repaired(q)
        for k in [0,1]:np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(a[2],b[2])
        olda=feature_view_spatial(RECIPE,r)[1];oldb=feature_view_spatial(RECIPE,q)[1]
        self.assertGreater(float(np.max(np.abs(olda-oldb))),1.)

    def test_frame_transform_is_bitwise_frozen(self):
        for i in range(3):
            r=spatial_record(i);r['spatial_support'][:3]=False
            a=feature_view_spatial(RECIPE,r);b=feature_view_spatial_repaired(r)
            np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[2],b[2]);self.assertEqual(a[1].shape,b[1].shape)

    def test_masked_statistics_match_observed_only_and_empty_is_zero(self):
        x=np.array([[1,4],[2,5],[100,6]],np.float32);mask=np.array([[1,0],[1,0],[0,0]],bool)
        a=masked_video_statistics(x,mask).reshape(4,2)
        np.testing.assert_array_equal(a[:,1],0);self.assertEqual(a[0,0],1.5);self.assertEqual(a[1,0],.5)
        np.testing.assert_allclose(a[2:,0],[1.1,1.9],rtol=0,atol=1e-6)
        with self.assertRaises(ValueError):masked_video_statistics(x,mask[:2])

    def test_labels_and_identity_not_read_by_repaired_features(self):
        r=spatial_record(0);q=copy.deepcopy(r);q.pop('labels');q.pop('sha256');q.pop('rgb');q.update(name='different')
        a=feature_view_spatial_repaired(r);b=feature_view_spatial_repaired(q)
        for k in [0,1]:np.testing.assert_array_equal(a[k],b[k])


@unittest.skipUnless(importlib.util.find_spec('sklearn'),'sklearn fitting environment')
class RepairedVideoFit(unittest.TestCase):
    def test_content_isolation_seed_and_portable_parity(self):
        rs=[spatial_record(i) for i in range(5)];rs[0]['labels'][:]=0
        with self.assertRaises(ValueError):fit_repaired_video(rs,[0,1,2],[2],7)
        vp,state=fit_repaired_video(rs,[0,1,2],[3],7);self.assertLessEqual(state['max_portable_error'],2e-6)
        rs[4]['labels'][:]=1;rs[4]['spatial']*=1000
        q,new=fit_repaired_video(rs,[0,1,2],[3],7);self.assertEqual(vp,q);self.assertEqual(state,new)

if __name__=='__main__':unittest.main()
