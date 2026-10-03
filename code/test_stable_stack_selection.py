import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import run_stable_stack_selection as selection
import numpy as np
from optimized_video_statistics import VideoTruth,video_statistics
from run_optimization_selection import Probabilities
from run_stable_stack_selection import select_risk_config,config_grid,cross_meta
from optimized_stack_v3 import BRANCHES

class StableSelectionTests(unittest.TestCase):
    def test_thirty_fixed_configs_and_no_veto(self):
        configs=list(config_grid());self.assertEqual(len(configs),30)
        for c in configs:self.assertEqual(c['video_threshold'],0);self.assertEqual(c['video_strength'],0)
    def test_excluded_outer_labels_never_consumed(self):
        records=[{'sha256':'n','labels':np.zeros(20),'fps':10.,'event_count':0}, {'sha256':'p','labels':np.r_[np.zeros(5),np.ones(10),np.zeros(5)],'fps':10.,'event_count':1}, {'labels':np.array([np.nan])}]
        p=Probabilities({0:np.full(20,.1),1:np.r_[np.full(5,.1),np.full(10,.8),np.full(5,.1)]},{0:.01,1:.01},None)
        selected,rows=select_risk_config(records,[0,1],p,2026);self.assertTrue(selected['risk_ceiling_feasible']);self.assertEqual(selected['stats'][3],1)
    def test_no_feasible_config_minimizes_relative_violation_before_score(self):
        records=[{'sha256':str(i),'labels':np.full(20,int(i>1)),'fps':10.,'event_count':int(i>1)} for i in range(8)]
        p=Probabilities({i:np.full(20,.8) for i in range(8)},{i:.5 for i in range(8)},None)
        selected,rows=select_risk_config(records,list(range(8)),p,2026)
        self.assertFalse(selected['risk_ceiling_feasible']);self.assertEqual(selected['risk_ceiling_relative_violation'],min(r['risk_ceiling_relative_violation'] for r in rows))
    def test_alias_repetition_cannot_change_content_risk_or_decoder_choice(self):
        records=[{'sha256':str(i),'labels':np.full(20,int(i>1)),'fps':10.,'event_count':int(i>1)} for i in range(6)]
        p=Probabilities({i:np.full(20,.05 if i in [0,1,2] else .8) for i in range(6)},{i:.5 for i in range(6)},None)
        original,rows=select_risk_config(records,list(range(6)),p,2026)
        records.append(dict(records[2]));p.frame[6]=p.frame[2];p.video[6]=p.video[2]
        alias,duplicated=select_risk_config(records,list(range(7)),p,2026)
        self.assertEqual(original['config'],alias['config'])
        for a,b in zip(rows,duplicated):
            self.assertAlmostEqual(a['stable_utility'],b['stable_utility'])
            self.assertAlmostEqual(a['inner_positive_empty_rate'],b['inner_positive_empty_rate'])
            self.assertEqual(a['risk_ceiling_feasible'],b['risk_ceiling_feasible'])
            np.testing.assert_allclose(a['content_stats'],b['content_stats'])

    def test_complete_fold_runs_subset_metrics_and_writes_unique_report(self):
        records=[{'sha256':str(i),'labels':np.r_[np.zeros(5),np.full(10,int(i%3!=0)),np.zeros(5)],'fps':10.,'event_count':int(i%3!=0)} for i in range(8)]
        train=list(range(6));val=[6,7];splits=[[0,1],[2,3],[4,5]]
        partitions=[{'fit':[i for i in train if i not in v],'validation':v} for v in splits]
        def loader(fold,recipe,rs):
            def pack(ids):return Probabilities({i:.1+.7*rs[i]['labels'] for i in ids},{i:.2+.6*bool(rs[i]['labels'].any()) for i in ids},None)
            return pack(train),pack(val),{'train':train,'validation':val,'inner_partitions':partitions}
        with tempfile.TemporaryDirectory() as directory,patch.object(selection,'_STATE',({'outer_folds':[val]},records,{})),patch.object(selection,'OUT',Path(directory)),patch.object(selection,'load_fold',side_effect=loader):
            selection.fold_task(0)
            self.assertTrue((Path(directory)/'folds/fold0.json').exists())
            with self.assertRaises(FileExistsError):selection.fold_task(0)

    def test_meta_rejects_duplicate_content_crossing(self):
        records=[{'sha256':'same','labels':np.zeros(20),'fps':10.,'event_count':0},{'sha256':'same','labels':np.ones(20),'fps':10.,'event_count':1}]
        pack={r:Probabilities({0:np.full(20,.2),1:np.full(20,.8)},{0:.2,1:.8},None) for b in BRANCHES for r,w in b}
        with self.assertRaises(ValueError):cross_meta(records,[0,1],pack,[{'fit':[1],'validation':[0]},{'fit':[0],'validation':[1]}],BRANCHES)

if __name__=='__main__':unittest.main()
