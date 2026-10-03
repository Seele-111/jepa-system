import unittest

import numpy as np

from jepa_graph_reranker import (
    build_proposal_graph,
    graph_rerank_candidate_scores,
    is_graph_reranker_improvement,
    select_graph_reranker_params,
    select_graph_reranked_predictions,
)


class JEPAGraphRerankerTests(unittest.TestCase):
    def test_proposal_graph_has_symmetric_support_and_parent_child_edges(self):
        candidates = [(0, 20), (2, 5), (30, 34)]
        nodes, edges = build_proposal_graph(candidates, np.array([0.8, 0.6, 0.4]), support_iou=0.15)
        self.assertEqual(len(nodes), 3)
        relations = {(edge.source, edge.target, edge.relation) for edge in edges}
        self.assertIn((0, 1, "contains"), relations)
        self.assertIn((0, 1, "support"), relations)
        self.assertIn((1, 0, "support"), relations)
        self.assertNotIn((0, 2, "support"), relations)

    def test_graph_scores_boost_consensus_candidate_over_isolated_candidate(self):
        candidates = [(10, 14), (10, 13), (11, 14), (30, 34)]
        scores = np.array([0.45, 0.44, 0.43, 0.45], dtype=np.float32)

        adjusted = graph_rerank_candidate_scores(
            mainline_segments=[],
            candidates=candidates,
            scores=scores,
            support_iou=0.3,
            support_weight=0.2,
            support_count_weight=0.1,
            split_penalty=0.0,
            mainline_overlap_penalty=0.0,
        )

        self.assertGreater(float(adjusted[0]), float(adjusted[3]))

    def test_graph_selection_prefers_supported_compact_children_over_merged_parent(self):
        candidates = [(0, 20), (1, 4), (10, 14), (1, 5), (9, 14)]
        scores = np.array([0.72, 0.62, 0.61, 0.59, 0.58], dtype=np.float32)

        selected = select_graph_reranked_predictions(
            mainline_predictions=[[]],
            candidate_predictions=[candidates],
            candidate_scores=[scores],
            prob_threshold=0.62,
            support_iou=0.3,
            support_weight=0.15,
            support_count_weight=0.05,
            split_penalty=0.25,
            mainline_overlap_penalty=0.0,
            length_penalty=0.0,
        )

        self.assertEqual(selected, [[(1, 4), (10, 14)]])

    def test_select_graph_reranker_params_enables_recall_gain_without_isolated_fp(self):
        mainline = [[(1, 2)]]
        candidates = [[(7, 8), (7, 9), (30, 31)]]
        scores = [np.array([0.45, 0.44, 0.45], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0] + [0] * 25, dtype=np.int64)]

        config, metrics, selector_predictions = select_graph_reranker_params(
            mainline,
            candidates,
            scores,
            labels,
            prob_thresholds=[0.5],
            support_ious=[0.3],
            support_weights=[0.2],
            support_count_weights=[0.1],
            split_penalties=[0.0],
            mainline_overlap_penalties=[0.0],
            length_penalties=[0.0],
            protected_mainline_iou_candidates=[0.0],
            selector_nms_iou_candidates=[None],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(selector_predictions, [[(7, 8)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)

    def test_graph_improvement_guard_requires_better_current_fusion_metric(self):
        current = {
            "segment": {"f1": 0.8, "precision": 0.9, "recall": 0.72},
            "frame": {"f1": 0.85},
        }
        graph_regression = {
            "segment": {"f1": 0.79, "precision": 0.95, "recall": 0.7},
            "frame": {"f1": 0.9},
        }
        graph_gain = {
            "segment": {"f1": 0.81, "precision": 0.88, "recall": 0.75},
            "frame": {"f1": 0.84},
        }

        self.assertFalse(is_graph_reranker_improvement(graph_regression, current))
        self.assertTrue(is_graph_reranker_improvement(graph_gain, current))


if __name__ == "__main__":
    unittest.main()
