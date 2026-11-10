import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.vector_store import VectorStore


class TestVectorStoreSnapshots(unittest.TestCase):
    def create_store(self):
        store = VectorStore()
        chunks = [
            {
                "chunk_id": "a-1",
                "document_id": "doc-a",
                "source": "alpha.pdf",
                "page": 1,
                "text": "Alpha document content.",
            },
            {
                "chunk_id": "b-1",
                "document_id": "doc-b",
                "source": "beta.pdf",
                "page": 1,
                "text": "Beta document content.",
            },
        ]
        store.add_chunks(chunks, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        store.register_document("doc-a", "alpha.pdf", 1, 1, 100)
        store.register_document("doc-b", "beta.pdf", 1, 1, 100)
        return store

    def current_snapshot(self, root: Path) -> Path:
        name = (root / "CURRENT").read_text(encoding="utf-8").strip()
        return root / "snapshots" / name

    def test_save_publishes_complete_snapshot_and_checksums(self):
        store = self.create_store()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store.save(root)

            current = self.current_snapshot(root)
            self.assertTrue((current / "manifest.json").exists())
            self.assertTrue((current / "index.faiss").exists())
            self.assertTrue((current / "embeddings.npy").exists())
            manifest = json.loads((current / "manifest.json").read_text())
            self.assertIn("chunks", manifest["files_sha256"])
            self.assertEqual(manifest["vector_count"], 2)

            loaded = VectorStore()
            self.assertTrue(loaded.load(root))
            self.assertEqual(loaded.count(), 2)
            self.assertEqual(loaded.document_count(), 2)

    def test_corrupt_current_snapshot_recovers_previous_snapshot(self):
        store = self.create_store()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store.save(root)
            previous_name = (root / "CURRENT").read_text().strip()

            updated = store.copy()
            updated.remove_document("doc-a")
            updated.save(root)
            current = self.current_snapshot(root)
            (current / "chunks.json").write_text("corrupt", encoding="utf-8")

            recovered = VectorStore()
            self.assertTrue(recovered.load(root))
            self.assertEqual(recovered.document_count(), 2)
            self.assertEqual(recovered.count(), 2)
            self.assertEqual((root / "CURRENT").read_text().strip(), previous_name)

    def test_checksum_detects_single_snapshot_corruption(self):
        store = self.create_store()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store.save(root)
            current = self.current_snapshot(root)
            (current / "documents.json").write_text("{}", encoding="utf-8")

            loaded = VectorStore()
            with self.assertRaisesRegex(ValueError, "No valid vector-store snapshot"):
                loaded.load(root)

    def test_legacy_flat_store_remains_loadable(self):
        store = self.create_store()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store.save(root)
            snapshot = self.current_snapshot(root)
            for filename in (
                "index.faiss",
                "embeddings.npy",
                "chunks.json",
                "documents.json",
                "manifest.json",
            ):
                shutil.copy2(snapshot / filename, root / filename)
            shutil.rmtree(root / "snapshots")
            (root / "CURRENT").unlink()

            loaded = VectorStore()
            self.assertTrue(loaded.load(root))
            self.assertEqual(loaded.count(), 2)
            loaded.save(root)
            self.assertTrue((root / "CURRENT").exists())

    def test_empty_directory_returns_false(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertFalse(VectorStore().load(temporary))


if __name__ == "__main__":
    unittest.main()
