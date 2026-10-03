import unittest
import importlib.util
import numpy as np
from optimized_event_training_v5 import RECIPES, event_weights, fit_event_member, export_event_member, predict_event_member
from test_optimized_compact_v4 import record

class EventTraining(unittest.TestCase):
    def test_short_and_long_events_have_equal_training_mass(self):
        a = record(0, 60); a['labels'][:] = 0; a['labels'][5:7] = 1; a['labels'][25:45] = 1
        b = record(1, 60); b['labels'][:] = 0
        w = event_weights([a, b], [0, 1])
        self.assertAlmostEqual(w[5:7].sum(), w[25:45].sum())
        y = np.r_[a['labels'], b['labels']]
        self.assertAlmostEqual(w[y > 0].sum(), w[y == 0].sum())
        self.assertAlmostEqual(w.mean(), 1)
        self.assertAlmostEqual(w[60:].sum() / w[:60][a['labels'] == 0].sum(), 2)

    def test_content_aliases_share_total_training_mass(self):
        a,b=record(0,40),record(1,40);b['labels'][:]=0
        w=event_weights([a,b],[0,1]);duplicate=event_weights([a,b,a],[0,1,2])
        # The final mean-one normalization scales with total frame count.
        np.testing.assert_allclose(duplicate[:40]+duplicate[80:],w[:40]*1.5)
        np.testing.assert_allclose(duplicate[40:80],w[40:]*1.5)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'requires existing WSL sklearn')
    def test_heldout_and_portable_parity(self):
        r=[record(i,60) for i in range(4)];r[0]['labels'][:]=0
        for recipe in RECIPES:
            f,v,s=fit_event_member(recipe,r,[0,1,2],[3],177)
            a,b=predict_event_member(export_event_member(s),r[3])
            np.testing.assert_allclose(a,f[3],atol=2e-6,rtol=0);self.assertAlmostEqual(b,v[3],delta=2e-6)
            with self.assertRaises(ValueError):fit_event_member(recipe,r,[0,1],[1,2],1)

if __name__=='__main__':unittest.main()
