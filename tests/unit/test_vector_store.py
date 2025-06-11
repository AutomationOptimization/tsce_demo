import types
import pytest
from tsce_agent_demo.utils import vector_store


def test_query_missing_index(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "FAISS", object())
    monkeypatch.setattr(vector_store, "OpenAIEmbeddings", lambda: None)
    with pytest.raises(FileNotFoundError):
        vector_store.query("q", index_dir=tmp_path / "nope")


def test_query_existing_index(tmp_path, monkeypatch):
    index_dir = tmp_path / "vec"
    index_dir.mkdir()

    dummy_store = types.SimpleNamespace(
        similarity_search_with_score=lambda text, k=5: ([types.SimpleNamespace(page_content="hit")], None)
    )

    class DummyFAISS:
        @staticmethod
        def load_local(path, embeddings):
            assert path == str(index_dir)
            return dummy_store

    monkeypatch.setattr(vector_store, "FAISS", DummyFAISS)
    monkeypatch.setattr(vector_store, "OpenAIEmbeddings", lambda: None)
    result = vector_store.query("q", index_dir=str(index_dir))
    assert result == ["hit"]
