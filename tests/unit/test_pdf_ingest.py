import types
import threading
import http.server
from contextlib import contextmanager

import pip._vendor.requests as real_requests
import tools.ingest.pdf_ingest as pdf_ingest


class DummyStore:
    def __init__(self):
        self.texts = []
        self.metas = []

    def add_texts(self, texts, metadatas):
        self.texts.extend(texts)
        self.metas.extend(metadatas)

    def save_local(self, path):
        self.saved = path


class DummyFAISS:
    def __init__(self, *a):
        self.store = DummyStore()

    def add_texts(self, *a, **kw):
        return self.store.add_texts(*a, **kw)

    def save_local(self, path):
        return self.store.save_local(path)


@contextmanager
def serve(code=200, data=b"file"):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(code)
            self.end_headers()
            if code == 200:
                self.wfile.write(data)

    httpd = http.server.HTTPServer(("localhost", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    try:
        yield f"http://localhost:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join()


def test_download_success(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_ingest, "OpenAIEmbeddings", lambda: types.SimpleNamespace(embed_query=lambda x: x, embedding_size=1))
    monkeypatch.setattr(pdf_ingest, "FAISS", DummyFAISS)
    monkeypatch.setattr(pdf_ingest.subprocess, "check_output", lambda *a, **kw: b"text")
    monkeypatch.setattr(pdf_ingest, "requests", real_requests)

    with serve() as base:
        paper = types.SimpleNamespace(url=f"{base}/p.pdf", title="t")
        pdf_ingest.ingest_papers([paper], index_dir=str(tmp_path))


def test_download_http_error(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(pdf_ingest, "OpenAIEmbeddings", lambda: types.SimpleNamespace(embed_query=lambda x: x, embedding_size=1))
    monkeypatch.setattr(pdf_ingest, "FAISS", DummyFAISS)
    monkeypatch.setattr(pdf_ingest.subprocess, "check_output", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not call")))
    monkeypatch.setattr(pdf_ingest, "requests", real_requests)

    with serve(404) as base:
        paper = types.SimpleNamespace(url=f"{base}/p.pdf", title="t")
        caplog.set_level("ERROR")
        pdf_ingest.ingest_papers([paper], index_dir=str(tmp_path))
        assert any("404" in r.message for r in caplog.records)
