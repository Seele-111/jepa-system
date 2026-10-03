"""Focused tests: true modality omission, frozen-reference identity and OOF selection."""
import copy
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
from test_optimized_compact_v4 import record
from optimized_compact_features_v4 import feature_view_v4, fit_rgb_pca
from optimized_feature_blocks_v10 import RECIPES, feature_view_blocks, fit_block_member, export_block_member, predict_block_member
from run_feature_blocks_v10 import partitions, select_inner, promotion_guards
from run_optimization_selection import Probabilities


class BlockFeatures(unittest.TestCase):
    def test_rich_transform_bitwise_same_as_frozen_v4(self):
        records=[record(i) for i in range(4)]; pca=fit_rgb_pca(records,[0,1,2])
        for r in records:
            a=feature_view_v4('compact_rgb_corrected_motion_et',r,pca)
            b=feature_view_blocks(RECIPES[-1],r,pca)
            for j in range(2):np.testing.assert_array_equal(a[j],b[j])
            self.assertEqual(a[2],b[2])

    def test_omitted_modalities_are_not_read(self):
        records=[record(i) for i in range(3)]; pca=fit_rgb_pca(records,[0,1])
        for recipe in RECIPES:
            r=copy.deepcopy(records[2]); q=copy.deepcopy(r)
            if 'corrected' not in recipe:
                q.pop('corrected');q.pop('corrected_names')
            if 'rgb' not in recipe:q.pop('rgb')
            a=feature_view_blocks(recipe,r,pca if 'rgb' in recipe else None)
            b=feature_view_blocks(recipe,q,pca if 'rgb' in recipe else None)
            for j in range(2):np.testing.assert_array_equal(a[j],b[j])
            self.assertEqual(a[2],b[2])
            self.assertEqual(any(n.startswith('corrected/') for n in a[2]),'corrected' in recipe)
            self.assertEqual(any(n.startswith('rgb/') for n in a[2]),'rgb' in recipe)

    def test_no_identity_or_labels_in_features(self):
        r=record();q=copy.deepcopy(r);q['labels'][:]=1;q.update(name='alias',sha256='different',generator='not a feature')
        for recipe in (RECIPES[0],RECIPES[2]):
            a=feature_view_blocks(recipe,r);b=feature_view_blocks(recipe,q)
            for j in range(2):np.testing.assert_array_equal(a[j],b[j])

    def test_invalid_recipe_and_fps(self):
        r=record()
        with self.assertRaises(ValueError):feature_view_blocks('unknown',r)
        r['fps']=0
        with self.assertRaises(ValueError):feature_view_blocks(RECIPES[0],r)

    def test_group_partitions_keep_aliases_together(self):
        rs=[record(i) for i in range(9)]
        for r in rs:r.update(event_count=1,generator='synthetic')
        rs.append(copy.deepcopy(rs[0]))
        ps=partitions(rs,list(range(len(rs))),2)
        self.assertEqual(sorted(i for p in ps for i in p['validation']),list(range(len(rs))))
        for p in ps:
            self.assertFalse({rs[i]['sha256'] for i in p['fit']} & {rs[i]['sha256'] for i in p['validation']})

    def test_shared_selector_rejects_extra_and_missing_evidence(self):
        p=Probabilities({0:np.zeros(60)},{0:.2},None)
        with self.assertRaises(ValueError):select_inner([record()],[0],{RECIPES[0]:p},0)
        probs={r:p for r in RECIPES};probs[RECIPES[0]]=Probabilities({0:np.zeros(60),1:np.zeros(60)},{0:.2},None)
        with self.assertRaises(ValueError):select_inner([record()],[0],probs,0)

    def test_shared_selector_only_receives_declared_inner_rows(self):
        p=Probabilities({0:np.zeros(60)},{0:.2},None);calls=[]
        def choose(records,ids,prob,seed):
            calls.append((ids,prob,seed));return {'stable_utility':.3,'key':[.3,.2,0,0,0,0], 'config':{'threshold':.5}}
        with patch('run_feature_blocks_v10.V8.choose',side_effect=choose):
            chosen,rows=select_inner([record()],[0],{r:p for r in RECIPES},5)
        self.assertEqual(chosen['candidate'],RECIPES[0]);self.assertEqual(len(calls),4)
        self.assertTrue(all(x[0]==[0] and x[1] is p and x[2]==20261007 for x in calls))


@unittest.skipUnless(importlib.util.find_spec('sklearn'),'sklearn environment required')
class BlockModels(unittest.TestCase):
    def records(self):
        rs=[record(i,frames=60) for i in range(6)]
        rs[0]['labels'][:]=0;rs[1]['labels'][:]=0
        return rs

    def test_rich_fit_exactly_same_as_v8_reference(self):
        from optimized_event_compact_v8 import fit_event_compact_member
        rs=self.records();a,b,_=fit_event_compact_member('event_compact_rgb_corrected_motion_et',rs,[0,1,2,3],[4,5],7)
        x,y,_=fit_block_member(RECIPES[-1],rs,[0,1,2,3],[4,5],7)
        for i in [4,5]:np.testing.assert_array_equal(a[i],x[i]);self.assertEqual(b[i],y[i])

    def test_leakage_rejected(self):
        rs=self.records();rs[5]['sha256']=rs[0]['sha256']
        with self.assertRaises(ValueError):fit_block_member(RECIPES[0],rs,[0,1,2],[5],3)
        with self.assertRaises(ValueError):fit_block_member(RECIPES[0],rs,[0,1,2],[3],3,full_fit=True)

    def test_portable_parity_all_blocks(self):
        rs=self.records()
        for recipe in RECIPES:
            fp,vp,state=fit_block_member(recipe,rs,[0,1,2,3],[4,5],9)
            member=export_block_member(state)
            for i in [4,5]:
                f,v=predict_block_member(member,rs[i])
                np.testing.assert_allclose(f,fp[i],rtol=0,atol=2e-6)
                self.assertAlmostEqual(v,vp[i],delta=2e-6)

if __name__=='__main__':unittest.main()
