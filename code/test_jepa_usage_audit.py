import tempfile
import textwrap
import unittest
import ast
from pathlib import Path

from jepa_usage_audit import audit_detect_pipeline


class JepaUsageAuditTests(unittest.TestCase):
    def _audit_source(self, source: str):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pipeline.py"
            path.write_text(textwrap.dedent(source), encoding="utf-8")
            return audit_detect_pipeline(path)

    def test_flags_encoder_distance_as_vjepa_proxy(self):
        findings = self._audit_source(
            """
            def extract_vjepa_tokens():
                encoder(batch)

            def score_vjepa_multi():
                cosine_similarity(a, b)

            def extract_ijepa_all():
                PatchEmbed()
                for blk in i_model.blocks:
                    x = blk(x)

            def score_ijepa_multi():
                fvar = pfeat.var(dim=1)
                spatial_inc = []
                energy = []
            """
        )

        vjepa = next(f for f in findings if f.component == "V-JEPA")
        ijepa = next(f for f in findings if f.component == "I-JEPA")
        self.assertEqual(vjepa.status, "proxy")
        self.assertEqual(ijepa.status, "proxy")

    def test_accepts_predictor_and_masks_as_objective_signal(self):
        findings = self._audit_source(
            """
            def extract_vjepa_tokens():
                encoder(batch)

            def score_vjepa_multi():
                predictor(tokens, masks_enc, masks_pred)

            def extract_ijepa_all():
                PatchEmbed()
                predictor(tokens, masks_enc, masks_pred)

            def score_ijepa_multi():
                pass
            """
        )

        vjepa = next(f for f in findings if f.component == "V-JEPA")
        ijepa = next(f for f in findings if f.component == "I-JEPA")
        self.assertEqual(vjepa.status, "objective")
        self.assertEqual(ijepa.status, "objective")

    def test_detect_report_defines_reproducible_jepa_usage_metadata(self):
        source = Path(__file__).with_name("detect_and_report_v4.py").read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        main_source = ast.get_source_segment(source, main_node)

        self.assertIn('"true_vjepa_predictor" if args.use_true_vjepa else "encoder_proxy"', main_source)
        self.assertIn('"referee_on_encoder_tokens"', main_source)
        self.assertIn('"predictor_embed_projection_proxy"', main_source)
        self.assertIn('"ijepa": "true_ijepa_predictor" if args.use_true_ijepa else "encoder_proxy"', main_source)
        self.assertIn('"jepa_usage": jepa_usage', main_source)


if __name__ == "__main__":
    unittest.main()
