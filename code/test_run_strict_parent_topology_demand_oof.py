import unittest
from types import SimpleNamespace

import numpy as np

from jepa_topology_matching_network import ChildSetProposal
from run_strict_parent_topology_demand_oof import (
    apply_voted_topology_predictions,
    run_strict_parent_topology_demand_oof,
)


class StrictParentTopologyDemandOOFTests(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            model="prototype",
            parent_min_lengths=[12],
            min_child_parent_coverages=[0.5],
            max_child_parent_ratios=[0.5],
            max_children_per_parent=[2],
            max_set_proposals_per_parent=[8],
            max_child_pool_per_parent=[8],
            thresholds=[0.0],
            risk_penalties=[0.25],
            safety_thresholds=[0.0],
            safety_weight=0.25,
            max_replaced_parents_per_video=[1],
            max_fp_increase=0,
            inference_vote_min_count=0,
            iou_threshold=0.3,
            seed=11,
            device="cpu",
            epochs=10,
            batch_size=16,
        )

    def _fold(self, fold_id: int):
        labels = np.zeros(24, dtype=np.int64)
        labels[1:5] = 1
        labels[11:15] = 1
        evidence = np.zeros((24, 3), dtype=np.float32)
        evidence[1:5] = 0.8
        evidence[11:15] = 0.75
        return {
            "fold": fold_id,
            "base": [[(0, 20)]],
            "candidates": [[(1, 4), (11, 14), (0, 20)]],
            "scores": [np.array([0.7, 0.65, 0.95], dtype=np.float32)],
            "evidence": [evidence],
            "labels": [labels],
        }

    def test_strict_oof_learns_parent_replacement_from_other_folds(self):
        result = run_strict_parent_topology_demand_oof([self._fold(0), self._fold(1), self._fold(2)], self._args())

        self.assertEqual(result["aggregate"]["segment"]["tp"], 6)
        self.assertEqual(result["aggregate"]["segment"]["fp"], 0)
        self.assertEqual(result["aggregate"]["segment"]["fn"], 0)
        for fold in result["folds"]:
            self.assertTrue(fold["config"]["enabled"])
            self.assertEqual(fold["predictions"], [[[1, 4], [11, 14]]])
            self.assertTrue(fold["config"]["inner_oof"])

    def test_apply_voted_topology_predictions_requires_consensus_votes(self):
        proposals = [
            ChildSetProposal(0, (0, 20), [(1, 4), (11, 14)], np.zeros(1, dtype=np.float32)),
            ChildSetProposal(0, (30, 50), [(31, 34), (41, 44)], np.zeros(1, dtype=np.float32)),
        ]

        predictions, diagnostics = apply_voted_topology_predictions(
            base_predictions=[[(0, 20), (30, 50)]],
            proposals=proposals,
            score_matrix=np.array([[0.9, 0.8], [0.92, 0.1]], dtype=np.float32),
            safety_matrix=np.array([[0.9, 0.9], [0.95, 0.1]], dtype=np.float32),
            threshold=0.5,
            safety_threshold=0.5,
            min_votes=2,
            max_replaced_parents_per_video=2,
        )

        self.assertEqual(predictions, [[(1, 4), (11, 14), (30, 50)]])
        self.assertEqual(diagnostics["accepted_proposals"], 1)
        self.assertEqual(diagnostics["max_votes"], 2)


if __name__ == "__main__":
    unittest.main()
