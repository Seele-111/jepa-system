"""Crossed fitter preserves train-only features, portable schema and fit parameters."""
import copy, importlib.util, unittest
import numpy as np
from optimized_event_compact_v8 import RECIPES, fit_event_compact_member, export_event_compact_member
from optimized_detector import predict_record, required_channels
from test_optimized_compact_v4 import record


class EventCompactTests(unittest.TestCase):
    def data(self):
        records=[record(i) for i in range(4)];records[0]['labels'][:]=0
        return records

    def test_content_overlap_is_rejected_before_fit(self):
        records=self.data();records[3]['sha256']=records[0]['sha256']
        with self.assertRaises(ValueError):fit_event_compact_member(RECIPES[0],records,[0,1,2],[3],101)
        with self.assertRaises(ValueError):fit_event_compact_member(RECIPES[0],records,[0,1],[3],101,full_fit=True)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'WSL sklearn runtime required')
    def test_both_recipes_portable_parameters_and_feature_schema(self):
        records=self.data()
        for recipe in RECIPES:
            fp,vp,state=fit_event_compact_member(recipe,records,[0,1,2],[3],101)
            self.assertEqual(state['fit_content_sha256'],sorted(r['sha256'] for r in records[:3]))
            self.assertEqual(state['training_recipe_id'],recipe)
            params=state['frame_model'].get_params()
            if recipe.endswith('_hgb'):
                self.assertEqual(params['max_iter'],180);self.assertEqual(params['min_samples_leaf'],24)
            else:self.assertEqual(params['n_estimators'],192);self.assertEqual(params['min_samples_leaf'],16)
            member=export_event_compact_member(state)
            self.assertEqual(required_channels(member),{'motion','local','corrected','rgb'})
            result=predict_record({'members':[member],'decoder':{'threshold':.5},'calibration':None},records[3])
            np.testing.assert_allclose(result['frame_probabilities'],fp[3],atol=2e-6,rtol=0)
            self.assertAlmostEqual(result['video_probability'],vp[3],delta=2e-6)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'WSL sklearn runtime required')
    def test_heldout_labels_cannot_change_training_or_prediction(self):
        records=self.data();a=fit_event_compact_member(RECIPES[0],records,[0,1,2],[3],101)
        changed=copy.deepcopy(records);changed[3]['labels'][:]=1-changed[3]['labels']
        b=fit_event_compact_member(RECIPES[0],changed,[0,1,2],[3],101)
        np.testing.assert_array_equal(a[0][3],b[0][3]);self.assertEqual(a[1][3],b[1][3])
        self.assertEqual(a[2]['transform'],b[2]['transform'])

if __name__=='__main__':unittest.main()
