import unittest

import numpy as np

from jepa_topology_matching_network import (
    ChildSetProposal,
    TOPOLOGY_MATCHING_FEATURE_NAMES,
    apply_topology_matching_predictions,
    build_topology_inference_records,
    build_topology_matching_records,
    enumerate_child_set_proposals,
    fit_topology_matching_risk_models,
    score_topology_matching_risk_records,
    select_topology_matching_params,
)


class JEPATopologyMatchingNetworkTests(unittest.TestCase):
    def test_coherence_guard_preserves_genuine_long_parent_when_residual_stays_active(self):
        features = np.zeros(len(TOPOLOGY_MATCHING_FEATURE_NAMES), dtype=np.float32)
        feature_idx = {name: idx for idx, name in enumerate(TOPOLOGY_MATCHING_FEATURE_NAMES)}
        features[feature_idx["evidence_island_coverage"]] = 0.98
        features[feature_idx["child_evidence_mass_fraction"]] = 0.52
        features[feature_idx["residual_active_fraction"]] = 0.82
        features[feature_idx["topology_count_alignment"]] = 0.5
        proposal = ChildSetProposal(
            video_idx=0,
            parent=(0, 20),
            children=[(1, 4), (11, 14)],
            features=features,
        )

        predictions = apply_topology_matching_predictions(
            [[(0, 20)]],
            [proposal],
            np.array([0.99], dtype=np.float32),
            threshold=0.5,
            max_replaced_parents_per_video=1,
            min_evidence_island_coverage=0.7,
            max_residual_active_fraction=0.25,
            min_topology_count_alignment=0.8,
        )

        self.assertEqual(predictions, [[(0, 20)]])

    def test_coherence_guard_allows_split_when_children_cover_islands_and_residual_is_quiet(self):
        features = np.zeros(len(TOPOLOGY_MATCHING_FEATURE_NAMES), dtype=np.float32)
        feature_idx = {name: idx for idx, name in enumerate(TOPOLOGY_MATCHING_FEATURE_NAMES)}
        features[feature_idx["evidence_island_coverage"]] = 0.98
        features[feature_idx["child_evidence_mass_fraction"]] = 0.96
        features[feature_idx["residual_active_fraction"]] = 0.02
        features[feature_idx["topology_count_alignment"]] = 1.0
        proposal = ChildSetProposal(
            video_idx=0,
            parent=(0, 20),
            children=[(1, 4), (11, 14)],
            features=features,
        )

        predictions = apply_topology_matching_predictions(
            [[(0, 20)]],
            [proposal],
            np.array([0.99], dtype=np.float32),
            threshold=0.5,
            max_replaced_parents_per_video=1,
            min_evidence_island_coverage=0.7,
            max_residual_active_fraction=0.25,
            min_topology_count_alignment=0.8,
        )

        self.assertEqual(predictions, [[(1, 4), (11, 14)]])

    def test_teacher_builds_complete_child_set_for_swallowed_parent(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14), (0, 20), (18, 19)]]
        scores = [np.array([0.35, 0.32, 0.9, 0.8], dtype=np.float32)]
        evidence = [np.zeros((24, 3), dtype=np.float32)]
        evidence[0][1:5] = 0.75
        evidence[0][11:15] = 0.72
        evidence[0][18:20] = 0.2
        labels = [
            np.array(
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.int64,
            )
        ]

        records = build_topology_matching_records(
            base,
            candidates,
            scores,
            evidence,
            labels,
            iou_threshold=0.3,
            parent_min_length=12,
            min_child_parent_coverage=0.5,
            max_child_parent_ratio=0.5,
            max_children_per_parent=3,
            max_set_proposals_per_parent=6,
            include_teacher=True,
        )

        positives = [record for record in records if record.label == 1]
        self.assertTrue(positives)
        self.assertIn([(1, 4), (11, 14)], [record.children for record in positives])
        self.assertGreater(max(record.split_tp - record.base_tp for record in positives), 0)

    def test_enumeration_keeps_separated_set_instead_of_duplicate_child(self):
        candidates = [(1, 4), (2, 5), (11, 14), (0, 20)]
        scores = np.array([0.9, 0.88, 0.55, 0.95], dtype=np.float32)
        evidence = np.zeros((24, 3), dtype=np.float32)
        evidence[1:5] = 0.8
        evidence[2:6] = 0.78
        evidence[11:15] = 0.7

        proposals = enumerate_child_set_proposals(
            parent=(0, 20),
            candidates=candidates,
            candidate_scores=scores,
            evidence=evidence,
            min_child_parent_coverage=0.5,
            max_child_parent_ratio=0.5,
            max_children_per_parent=3,
            max_set_proposals=8,
            nms_iou=0.3,
        )

        self.assertIn([(1, 4), (11, 14)], [proposal.children for proposal in proposals])
        self.assertNotIn([(1, 4), (2, 5)], [proposal.children for proposal in proposals])

    def test_v2_features_capture_evidence_island_demand_and_low_residual(self):
        candidates = [(1, 4), (11, 14), (0, 20), (18, 19)]
        scores = np.array([0.7, 0.65, 0.95, 0.2], dtype=np.float32)
        evidence = np.zeros((24, 3), dtype=np.float32)
        evidence[1:5] = 0.8
        evidence[11:15] = 0.75

        proposals = enumerate_child_set_proposals(
            parent=(0, 20),
            candidates=candidates,
            candidate_scores=scores,
            evidence=evidence,
            min_child_parent_coverage=0.5,
            max_child_parent_ratio=0.5,
            max_children_per_parent=2,
            max_set_proposals=8,
            nms_iou=0.3,
        )
        target = [proposal for proposal in proposals if proposal.children == [(1, 4), (11, 14)]][0]
        feature = dict(zip(TOPOLOGY_MATCHING_FEATURE_NAMES, target.features))

        self.assertEqual(feature["evidence_island_count"], 2.0)
        self.assertGreater(feature["evidence_island_coverage"], 0.95)
        self.assertGreater(feature["child_evidence_mass_fraction"], 0.95)
        self.assertLess(feature["residual_active_fraction"], 0.05)
        self.assertAlmostEqual(feature["topology_count_alignment"], 1.0)
        self.assertGreater(feature["set_evidence_demand_score"], 2.5)

    def test_build_topology_inference_records_does_not_require_labels(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14), (0, 20)]]
        scores = [np.array([0.7, 0.65, 0.95], dtype=np.float32)]
        evidence = [np.zeros((24, 3), dtype=np.float32)]
        evidence[0][1:5] = 0.8
        evidence[0][11:15] = 0.75

        records = build_topology_inference_records(
            base,
            candidates,
            scores,
            evidence,
            parent_min_length=12,
            min_child_parent_coverage=0.5,
            max_child_parent_ratio=0.5,
            max_children_per_parent=2,
            max_set_proposals_per_parent=8,
            max_child_pool_per_parent=8,
        )

        self.assertTrue(records)
        self.assertIn([(1, 4), (11, 14)], [record.children for record in records])
        self.assertTrue(all(record.label == 0 for record in records))
        self.assertTrue(all(record.gain_label == 0 for record in records))
        self.assertTrue(all(record.safe_label == 0 for record in records))

    def test_enumeration_limits_child_pool_before_combinations(self):
        candidates = [(idx, idx) for idx in range(30)]
        scores = np.linspace(1.0, 0.1, num=30, dtype=np.float32)
        evidence = np.zeros((40, 3), dtype=np.float32)

        proposals = enumerate_child_set_proposals(
            parent=(0, 39),
            candidates=candidates,
            candidate_scores=scores,
            evidence=evidence,
            min_child_parent_coverage=0.5,
            max_child_parent_ratio=0.5,
            max_children_per_parent=3,
            max_set_proposals=1000,
            nms_iou=0.3,
            max_child_pool=5,
        )

        self.assertLessEqual(len(proposals), 20)
        used_children = {child for proposal in proposals for child in proposal.children}
        self.assertTrue(all(child[0] < 5 for child in used_children))

    def test_select_params_replaces_parent_with_learned_matching_set(self):
        base = [[(0, 20)], [(0, 20)]]
        candidates = [
            [(1, 4), (11, 14), (0, 20), (18, 19)],
            [(1, 4), (11, 14), (0, 20), (18, 19)],
        ]
        scores = [
            np.array([0.35, 0.32, 0.9, 0.8], dtype=np.float32),
            np.array([0.36, 0.33, 0.91, 0.81], dtype=np.float32),
        ]
        evidence = [np.zeros((24, 3), dtype=np.float32), np.zeros((24, 3), dtype=np.float32)]
        for item in evidence:
            item[1:5] = 0.75
            item[11:15] = 0.72
            item[18:20] = 0.2
        labels = [
            np.array(
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.int64,
            ),
            np.array(
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.int64,
            ),
        ]

        config, metrics, predictions = select_topology_matching_params(
            train_base_predictions=base[:1],
            train_candidate_predictions=candidates[:1],
            train_candidate_scores=scores[:1],
            train_evidence_arrays=evidence[:1],
            train_labels=labels[:1],
            val_base_predictions=base[1:],
            val_candidate_predictions=candidates[1:],
            val_candidate_scores=scores[1:],
            val_evidence_arrays=evidence[1:],
            val_labels=labels[1:],
            model_name="prototype",
            seed=7,
            thresholds=[0.0, 0.25, 0.5],
            parent_min_lengths=[12],
            min_child_parent_coverages=[0.5],
            max_child_parent_ratios=[0.5],
            max_children_per_parents=[3],
            max_set_proposals_per_parents=[6],
            max_child_pool_per_parents=[8],
            max_replaced_parents_per_videos=[1],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(predictions, [[(1, 4), (11, 14)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)

    def test_disabled_selection_reports_diagnostics(self):
        base = [[(0, 20)], [(0, 20)]]
        candidates = [[(1, 4), (11, 14), (0, 20)], [(1, 4), (11, 14), (0, 20)]]
        scores = [
            np.array([0.5, 0.4, 0.9], dtype=np.float32),
            np.array([0.5, 0.4, 0.9], dtype=np.float32),
        ]
        evidence = [np.zeros((24, 3), dtype=np.float32), np.zeros((24, 3), dtype=np.float32)]
        labels = [np.zeros(24, dtype=np.int64), np.zeros(24, dtype=np.int64)]

        config, _, _ = select_topology_matching_params(
            train_base_predictions=base[:1],
            train_candidate_predictions=candidates[:1],
            train_candidate_scores=scores[:1],
            train_evidence_arrays=evidence[:1],
            train_labels=labels[:1],
            val_base_predictions=base[1:],
            val_candidate_predictions=candidates[1:],
            val_candidate_scores=scores[1:],
            val_evidence_arrays=evidence[1:],
            val_labels=labels[1:],
            model_name="prototype",
            seed=7,
            thresholds=[0.2],
            parent_min_lengths=[12],
            min_child_parent_coverages=[0.5],
            max_child_parent_ratios=[0.7],
            max_children_per_parents=[2],
            max_set_proposals_per_parents=[4],
            max_child_pool_per_parents=[8],
            max_replaced_parents_per_videos=[1],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertFalse(config["enabled"])
        self.assertGreater(config["diagnostics"]["configs_considered"], 0)
        self.assertGreater(config["diagnostics"]["train_single_class"], 0)

    def test_fp_rejected_selection_reports_best_rejected_candidate(self):
        base = [[(0, 20)], [(0, 14)]]
        candidates = [
            [(1, 4), (11, 14), (18, 19), (0, 20)],
            [(1, 4), (12, 13), (0, 14)],
        ]
        scores = [
            np.array([0.35, 0.32, 0.9, 0.95], dtype=np.float32),
            np.array([0.36, 0.91, 0.96], dtype=np.float32),
        ]
        evidence = [np.zeros((24, 3), dtype=np.float32), np.zeros((24, 3), dtype=np.float32)]
        for item in evidence:
            item[1:5] = 0.75
            item[11:15] = 0.72
            item[18:20] = 0.9
        labels = [
            np.array(
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.int64,
            ),
            np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.int64,
            ),
        ]

        config, _, _ = select_topology_matching_params(
            train_base_predictions=base[:1],
            train_candidate_predictions=candidates[:1],
            train_candidate_scores=scores[:1],
            train_evidence_arrays=evidence[:1],
            train_labels=labels[:1],
            val_base_predictions=base[1:],
            val_candidate_predictions=candidates[1:],
            val_candidate_scores=scores[1:],
            val_evidence_arrays=evidence[1:],
            val_labels=labels[1:],
            model_name="prototype",
            seed=7,
            thresholds=[0.0],
            parent_min_lengths=[12],
            min_child_parent_coverages=[0.5],
            max_child_parent_ratios=[0.5],
            max_children_per_parents=[3],
            max_set_proposals_per_parents=[4],
            max_child_pool_per_parents=[8],
            max_replaced_parents_per_videos=[1],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertFalse(config["enabled"])
        self.assertGreater(config["diagnostics"]["fp_rejected_configs"], 0)
        self.assertIsNotNone(config["diagnostics"]["best_rejected_metrics"])

    def test_risk_model_separates_gain_from_fp_increase(self):
        safe_feature = np.zeros(27, dtype=np.float32)
        safe_feature[2] = 2.0
        safe_feature[6] = 0.45
        safe_feature[16] = 0.8
        risky_feature = safe_feature.copy()
        risky_feature[2] = 3.0
        risky_feature[6] = 0.9
        risky_feature[22] = 0.75

        records = [
            build_topology_matching_records(
                [[(0, 20)]],
                [[(1, 4), (11, 14), (0, 20)]],
                [np.array([0.7, 0.65, 0.95], dtype=np.float32)],
                [np.zeros((24, 3), dtype=np.float32)],
                [
                    np.array(
                        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        dtype=np.int64,
                    )
                ],
                iou_threshold=0.3,
                parent_min_length=12,
                min_child_parent_coverage=0.5,
                max_child_parent_ratio=0.5,
                max_children_per_parent=2,
                max_set_proposals_per_parent=2,
                include_teacher=True,
            )[0],
            build_topology_matching_records(
                [[(0, 20)]],
                [[(1, 4), (11, 14), (18, 19), (0, 20)]],
                [np.array([0.7, 0.65, 0.95, 0.99], dtype=np.float32)],
                [np.zeros((24, 3), dtype=np.float32)],
                [
                    np.array(
                        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        dtype=np.int64,
                    )
                ],
                iou_threshold=0.3,
                parent_min_length=12,
                min_child_parent_coverage=0.5,
                max_child_parent_ratio=0.5,
                max_children_per_parent=3,
                max_set_proposals_per_parent=8,
                include_teacher=False,
            )[-1],
        ]
        records[0] = records[0].__class__(
            **{**records[0].__dict__, "features": safe_feature, "gain_label": 1, "safe_label": 1, "label": 1}
        )
        records[1] = records[1].__class__(
            **{**records[1].__dict__, "features": risky_feature, "gain_label": 1, "safe_label": 0, "label": 0}
        )

        gain_model, safety_model = fit_topology_matching_risk_models(
            records,
            model_name="prototype",
            seed=11,
        )
        combined, gain_scores, safety_scores = score_topology_matching_risk_records(
            gain_model,
            safety_model,
            records,
            risk_penalty=0.5,
        )

        self.assertGreater(gain_scores[1], 0.5)
        self.assertGreater(safety_scores[0], safety_scores[1])
        self.assertGreater(combined[0], combined[1])


if __name__ == "__main__":
    unittest.main()
