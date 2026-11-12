"""Suíte de testes automatizados (unitários e de integração) do sistema multimodal."""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import COLLECTION, META_PATH, PROCESSED_DIR  # noqa: E402
from app.services.fusion_service import rrf  # noqa: E402
from ingestion.load_catalog import (  # noqa: E402
    build_text_r2,
    build_text_r4,
    load_catalog,
    point_uuid,
)


class TestRRF(unittest.TestCase):
    def test_rrf_combines_rankings(self):
        r1 = ["item_a", "item_b", "item_c"]
        r2 = ["item_b", "item_a", "item_d"]
        fused = rrf([r1, r2], k=60)
        ids = [item_id for item_id, _ in fused]
        # item_a e item_b aparecem no topo de ambas as listas (1/61 + 1/62)
        self.assertEqual(set(ids[:2]), {"item_a", "item_b"})
        self.assertAlmostEqual(fused[0][1], 1 / 61 + 1 / 62, places=6)

    def test_rrf_empty(self):
        self.assertEqual(rrf([]), [])
        self.assertEqual(rrf([[]]), [])


class TestCatalogRepresentations(unittest.TestCase):
    def test_deterministic_point_uuid(self):
        u1 = point_uuid("MN_003_14231_001")
        u2 = point_uuid("MN_003_14231_001")
        u3 = point_uuid("MN_003_14231_002")
        self.assertEqual(u1, u2)
        self.assertNotEqual(u1, u3)

    def test_r2_contains_only_physical_description(self):
        rec = {
            "descricao_catalogo": "Cesto trançado de palha com alça.",
            "categoria": "trançado",
            "povo": "Tikuna",
            "material": "fibra vegetal",
        }
        r2 = build_text_r2(rec)
        self.assertEqual(r2, "Cesto trançado de palha com alça.")
        self.assertNotIn("Tikuna", r2)
        self.assertNotIn("Categoria:", r2)

    def test_r4_includes_selected_metadata(self):
        rec = {
            "descricao_catalogo": "Cesto trançado de palha com alça.",
            "categoria": "trançado",
            "identificacao_local": "paneiro",
            "material": "fibra vegetal",
            "tecnica": "trançado",
            "funcao": "transporte",
            "povo": "Tikuna",
            "auto_denominacao": None,
            "aldeia": "Umariaçu",
            "notas_gerais": None,
        }
        r4 = build_text_r4(rec)
        self.assertIn("Cesto trançado de palha com alça.", r4)
        self.assertIn("Categoria: trançado.", r4)
        self.assertIn("Denominação local/indígena: paneiro.", r4)
        self.assertIn("Material e suporte: fibra vegetal.", r4)
        self.assertIn("Povo: Tikuna.", r4)

    def test_load_catalog_integrates_real_dataset(self):
        records = load_catalog()
        self.assertEqual(len(records), 113)
        items = {r["item_id"] for r in records}
        self.assertEqual(len(items), 35)
        for r in records:
            self.assertTrue(r["texto_r2"])
            self.assertTrue(r["texto_r4"])
            self.assertGreaterEqual(len(r["texto_r4"]), len(r["texto_r2"]))


class TestEvaluationMetrics(unittest.TestCase):
    def test_perfect_ranking_metrics(self):
        try:
            from evaluation.evaluate import average_precision, metrics
        except ImportError:
            self.skipTest("Dependências do SearchService indisponíveis neste interpretador")
        ranked = ["A", "B", "C", "D"]
        relevant = {"A", "B"}
        m = metrics(ranked, relevant)
        self.assertAlmostEqual(m["precision@1"], 1.0)
        self.assertAlmostEqual(m["recall@5"], 1.0)
        self.assertAlmostEqual(m["ndcg@5"], 1.0)
        self.assertAlmostEqual(m["mrr"], 1.0)
        self.assertAlmostEqual(average_precision(ranked, relevant, k=10), 1.0)

    def test_missed_first_hit_metrics(self):
        try:
            from evaluation.evaluate import metrics
        except ImportError:
            self.skipTest("Dependências do SearchService indisponíveis neste interpretador")
        ranked = ["X", "A", "Y"]
        relevant = {"A"}
        m = metrics(ranked, relevant)
        self.assertAlmostEqual(m["recall@1"], 0.0)
        self.assertAlmostEqual(m["recall@5"], 1.0)
        self.assertAlmostEqual(m["mrr"], 0.5)
        self.assertAlmostEqual(m["ndcg@5"], 1 / math.log2(3), places=5)


class TestBM25AndConfig(unittest.TestCase):
    def test_config_paths_exist(self):
        self.assertTrue(PROCESSED_DIR.exists())
        self.assertTrue(META_PATH.exists())
        self.assertEqual(COLLECTION, "museum_images")

    def test_bm25_lexical_retrieval_and_filters(self):
        from app.services.bm25_service import BM25Index, _tokenize_pt

        tokens = _tokenize_pt("Cestos trançados utilizados para transporte de frutas")
        self.assertIn("cesto", tokens)
        self.assertIn("trancado", tokens)
        self.assertNotIn("para", tokens)

        docs = {
            "1": {"item_id": "1", "categoria": "trançado", "texto_r2": "Cesto cargueiro de fibra vegetal."},
            "2": {"item_id": "2", "categoria": "cerâmica", "texto_r2": "Pote de barro cozido para água."},
        }
        idx = BM25Index(docs, "texto_r2")
        hits = idx.search("cesto de fibra", top_k=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0]["item_id"], "1")

        # Filtro por categoria deve restringir resultados
        filtered = idx.search("cesto de fibra", top_k=5, filters={"categoria": "cerâmica"})
        self.assertEqual(len(filtered), 0)


if __name__ == "__main__":
    unittest.main()
