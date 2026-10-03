"""Focused boundary tests: identity, endpoint-local acceptance and train isolation."""
import copy,importlib.util,unittest
import numpy as np
from test_optimized_compact_v4 import record
from optimized_compact_boundary_v12 import apply_boundary,configurations,fit_content_boundary,choose_boundary
from optimized_feature_blocks_v10 import feature_view_blocks,RECIPES
from optimized_boundary_head import predict_boundary_heads
from run_optimization_selection import Probabilities


class BoundaryPolicy(unittest.TestCase):
    def test_control_is_identity_and_never_forces_an_event(self):
        heads=(np.zeros(20,np.float32),np.zeros(20,np.float32));policy={'boundary_seconds':0.,'minimum_boundary_evidence':0.}
        self.assertEqual(apply_boundary([(2,8),(11,15)],heads,10.,policy)[0],[(2,8),(11,15)])
        self.assertEqual(apply_boundary([],heads,10.,policy)[0],[])

    def test_endpoint_peaks_refine_and_verify(self):
        s=np.zeros(20,np.float32);e=s.copy();s[3]=.8;e[9]=.7
        policy={'boundary_seconds':.2,'minimum_boundary_evidence':.5}
        pred,trace=apply_boundary([(2,10)],(s,e),10.,policy)
        self.assertEqual(pred,[(3,9)]);self.assertTrue(trace[0]['accepted'])
        e[9]=.2
        self.assertEqual(apply_boundary([(2,10)],(s,e),10.,policy)[0],[])

    def test_bounded_grid_and_bad_cutoff(self):
        self.assertEqual(len(list(configurations())),10)
        with self.assertRaises(ValueError):apply_boundary([], (np.zeros(3),np.zeros(3)),16.,{'boundary_seconds':.2,'minimum_boundary_evidence':float('nan')})

    def test_unseen_labels_cannot_change_inner_selection(self):
        rs=[record(i) for i in range(5)]
        for r in rs:r.update(event_count=1,generator='synthetic')
        rs[0]['labels'][:]=0;rs[0]['event_count']=0
        ids=[0,1,2,3];p=Probabilities({i:np.linspace(.1,.8,60).astype(np.float32) for i in ids},{i:.8 for i in ids},None)
        heads={i:(np.full(60,.2,np.float32),np.full(60,.2,np.float32)) for i in ids}
        a=choose_boundary(rs,ids,p,heads,0);rs[4]['labels'][:]=1;rs[4]['event_count']=8
        self.assertEqual(a,choose_boundary(rs,ids,p,heads,0))
        heads[4]=(np.zeros(60),np.zeros(60))
        with self.assertRaises(ValueError):choose_boundary(rs,ids,p,heads,0)


@unittest.skipUnless(importlib.util.find_spec('sklearn'),'sklearn environment required')
class BoundaryFit(unittest.TestCase):
    def records(self):
        rs=[record(i) for i in range(5)];rs[0]['labels'][:]=0;return rs

    def test_content_leakage_and_wrong_fullfit_rejected(self):
        rs=self.records();rs[4]['sha256']=rs[0]['sha256']
        with self.assertRaises(ValueError):fit_content_boundary(rs,[0,1,2],[4],3)
        with self.assertRaises(ValueError):fit_content_boundary(rs,[0,1,2],[3],3,full_fit=True)

    def test_pca_and_heads_fit_content_only_and_portable(self):
        rs=self.records();prob,state=fit_content_boundary(rs,[0,1,2],[3],3)
        rs[4]['labels'][:]=1;rs[4]['rgb']*=1000
        again,new=fit_content_boundary(rs,[0,1,2],[3],3)
        self.assertEqual(state,new)
        x,_,_=feature_view_blocks(RECIPES[-1],rs[3],state['transform']['pca'])
        expected=predict_boundary_heads(state['boundary_models'],x)
        for a,b in zip(prob[3],expected):np.testing.assert_array_equal(a,b)
        self.assertEqual(state['fit_content_sha256'],['content0','content1','content2'])

    def test_alias_total_content_mass_preserved(self):
        rs=self.records();rs.append(copy.deepcopy(rs[1]))
        _,state=fit_content_boundary(rs,[0,1,2,5],[3],3)
        self.assertEqual(state['base_content_mass'],3.)

if __name__=='__main__':unittest.main()
