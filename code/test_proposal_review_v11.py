"""Focused tests for proposal-local review: no global veto or score multiplication."""
import copy
import unittest
import numpy as np
from optimized_proposal_review_v11 import KIND, verify_proposals, choose_review, review_configurations
from run_optimization_selection import Probabilities
from test_optimized_compact_v4 import record


def cfg(minimum=.45,rescue=None):return {'kind':KIND,'minimum':minimum,'rescue_peak':rescue}


class ProposalReview(unittest.TestCase):
    def test_reviewer_reads_only_candidate_and_preserves_spans(self):
        p=np.full(12,.6);q=np.ones(12);q[3:6]=.2
        accepted,trace=verify_proposals([(3,5)],p,q,cfg())
        self.assertEqual(accepted,[])
        q[:3]=0;q[6:]=0
        self.assertEqual(verify_proposals([(3,5)],p,q,cfg())[0],accepted)
        q[3:6]=.6
        self.assertEqual(verify_proposals([(3,5)],p,q,cfg())[0],[(3,5)])

    def test_control_is_identity_even_for_low_reviewer(self):
        spans=[(0,1),(4,8)]
        accepted,trace=verify_proposals(spans,np.full(10,.4),np.zeros(10),cfg(0))
        self.assertEqual(accepted,spans);self.assertTrue(all(t['accepted'] for t in trace))

    def test_strong_local_peak_rescue(self):
        p=np.full(10,.5);p[5]=.85;q=np.zeros(10)
        accepted,trace=verify_proposals([(0,2),(4,6)],p,q,cfg(.45,.8))
        self.assertEqual(accepted,[(4,6)]);self.assertTrue(trace[1]['rescued'])
        self.assertEqual(verify_proposals([(0,2),(4,6)],p,q,cfg(.45))[0],[])

    def test_invalid_input_fails_closed(self):
        p=np.zeros(10)
        for spans,q,c in [([(0,10)],p,cfg()), ([(True,3)],p,cfg()), ([(2,1)],p,cfg()),
            ([(0,2)],np.zeros(9),cfg()), ([(0,2)],p,cfg(float('nan'))), ([(0,2)],p,{'kind':'bad','minimum':.4,'rescue_peak':None})]:
            with self.assertRaises(ValueError):verify_proposals(spans,p,q,c)
        q=p.copy();q[2]=float('nan')
        with self.assertRaises(ValueError):verify_proposals([(0,2)],p,q,cfg())

    def test_empty_proposals_and_bounded_grid(self):
        self.assertEqual(verify_proposals([],np.zeros(3),np.zeros(3),cfg())[0],[])
        policies=list(review_configurations());self.assertEqual(len(policies),7)
        self.assertEqual(policies[0],cfg(0))

    def test_selection_cannot_use_outer_labels_or_extra_scores(self):
        rs=[record(i) for i in range(5)]
        for r in rs:r.update(event_count=1,generator='synthetic')
        rs[0]['labels'][:]=0;rs[0]['event_count']=0
        ids=[0,1,2,3];fp={i:np.linspace(.1,.8,60).astype(np.float32) for i in ids}
        p=Probabilities(fp,{i:.8 for i in ids},None);q=Probabilities({i:v*.8 for i,v in fp.items()},{i:.5 for i in ids},None)
        a=choose_review(rs,ids,p,q,0)
        rs[4]['labels'][:]=1;rs[4]['event_count']=9;rs[4]['sha256']='ignored outer identity'
        b=choose_review(rs,ids,p,q,0)
        self.assertEqual(a,b)
        extra=Probabilities(dict(fp,**{}),dict(p.video),None);extra.frame[4]=np.zeros(60)
        with self.assertRaises(ValueError):choose_review(rs,ids,extra,q,0)

if __name__=='__main__':unittest.main()
