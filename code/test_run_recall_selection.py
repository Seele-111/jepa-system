import unittest
import numpy as np
from optimized_grouped_training import grouped_folds
from run_recall_selection import choose,evaluate,grouped_bootstrap,calibrate,select_legacy_family
from run_optimization_selection import Probabilities
from run_algorithm_optimization import choose_decoder
from optimized_locator import metrics

class GroupedSelectionTests(unittest.TestCase):
    def test_legacy_family_tie_preserves_order_not_decoder_secondary_metrics(self):
        first={'candidate':'motion_rf','key':(0.,0.,0.,0,-200)}
        second={'candidate':'motion_et','key':(0.,1.,1.,0,0)}
        self.assertIs(select_legacy_family([first,second]),first)
        second['key']=(.001,0.,0.,0,-2000)
        self.assertIs(select_legacy_family([first,second]),second)

    def test_identical_content_cannot_cross_any_fold(self):
        records=[{'sha256':str(k//2),'generator':'x','event_count':1} for k in range(12)]
        folds=grouped_folds(records,list(range(12)),3,99)
        for a,f in enumerate(folds):
            for b,g in enumerate(folds[:a]):self.assertFalse({records[i]['sha256'] for i in f}&{records[i]['sha256'] for i in g})
        self.assertEqual(sorted(sum(folds,[])),list(range(12)))
    def test_stats_search_matches_legacy_full_evaluator_exactly(self):
        rng=np.random.default_rng(31);records=[];fp={};vp={}
        for i in range(5):
            y=np.zeros(20);y[5:12]=int(i>0)
            records.append({'labels':y,'fps':10.,'event_count':int(i>0),'sha256':str(i),'generator':'x'})
            fp[i]=rng.random(20).astype(np.float32);vp[i]=float(rng.random())
        p=Probabilities(fp,vp,None);config,result=choose_decoder(records,list(range(5)),fp,vp)
        faster=choose(records,list(range(5)),p,False,99)
        self.assertEqual(config,faster['config'])
        self.assertEqual(result,metrics(list(evaluate(records,list(range(5)),p,config).values()),[r['labels'] for r in records]))
    def test_selector_rejects_unknown_or_incomplete_calibration(self):
        p=Probabilities({0:np.array([.2,.8],np.float32)},{0:.8},None)
        for state in [{'kind':'typo_v9'},{'slope':2.,'intercept':1.},{},False,{'frame':None}]:
            with self.subTest(state=state),self.assertRaises(ValueError):calibrate(p,state)
        self.assertIs(calibrate(p,None),p)

    def test_grouped_bootstrap_keeps_duplicate_alias_together(self):
        records=[{'sha256':'same' if i in [0,1] else str(i),'event_count':2 if i<2 else 0} for i in range(5)]
        counts=grouped_bootstrap(records,list(range(5)),991)
        np.testing.assert_array_equal(counts[:,0],counts[:,1])
        np.testing.assert_array_equal(counts[:,2:].sum(1),3)
    def test_v2_selection_does_not_consume_excluded_labels(self):
        records=[{'labels':np.zeros(12),'fps':10.,'event_count':0,'sha256':'normal'}, {'labels':np.ones(12),'fps':10.,'event_count':1,'sha256':'positive'}, {'labels':np.array([np.nan]),'fps':10.,'event_count':1,'sha256':'outer'}]
        p=Probabilities({0:np.full(12,.2,np.float32),1:np.full(12,.8,np.float32)},{0:.2,1:.8},None)
        selected=choose(records,[0,1],p,True,55,joint=True)
        self.assertEqual(evaluate(records,[0,1],p,selected['config']),{0:[],1:[(0,11)]})

if __name__=='__main__':unittest.main()
