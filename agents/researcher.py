import os
import requests
from typing import List

from tsce_agent_demo.tsce_chat import TSCEChat

from tools import (
    GoogleSearch,
    WebScrape,
    CreateFileTool,
    ReadFileTool,
    EditFileTool,
    DeleteFileTool,
    RunScriptTool,
)

from .base import BaseAgent
from .base_agent import compose_sections

class Researcher(BaseAgent):
    """Agent capable of searching the web and reading/writing files."""

    def __init__(self, model: str | None = None, *, log_dir: str | None = None) -> None:
        model_name = model or os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self._chat = TSCEChat(model=model_name)
        super().__init__(name="Researcher", chat=self._chat, log_dir=log_dir)
        del self.chat
        self.system_message = (
            "You are a meticulous research assistant. "
            "Use your search and file tools when helpful."
        )
        self.model = model_name
        self.history: List[str] = []
        self.search_tool = GoogleSearch()
        self.scrape_tool = WebScrape()
        self.create_tool = CreateFileTool()
        self.read_tool = ReadFileTool()
        self.edit_tool = EditFileTool()
        self.delete_tool = DeleteFileTool()
        self.run_tool = RunScriptTool()

    def chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        reply = self._chat(messages)
        return reply.content

    # ------------------------------------------------------------------
    def send_message(self, message: str) -> str:
        """Send ``message`` to the model and return the reply."""
        reply = self.chat(message)
        formatted = compose_sections("", "", reply)
        self.history.append(message)
        self.history.append(formatted)
        return formatted

    # Convenience wrappers -------------------------------------------------
    def search(self, query: str, *, backend: str | None = None):
        """Return search results for ``query`` using the selected backend."""
        backend = backend or os.getenv("SEARCH_BACKEND", "google").lower()
        if backend == "opensearch":
            from opensearchpy import OpenSearch
            from tools import embed_text

            host = os.getenv("OPENSEARCH_HOST", "localhost")
            index = os.getenv("OPENSEARCH_INDEX", "pubmed_fibrosis")
            client = OpenSearch([{"host": host, "port": 9200}])
            vec = embed_text(query)
            body = {
                "size": 5,
                "query": {"knn": {"embedding": {"vector": vec, "k": 5}}},
            }
            resp = client.search(index=index, body=body)
            result = [h["_source"]["abstract"] for h in resp["hits"]["hits"]]
        else:
            result = self.search_tool(query)

        self.history.append(f"search({query!r}) -> {result!r}")
        return result

    def read_file(self, path: str) -> str:
        result = self.read_tool(path)
        self.history.append(f"read_file({path!r}) -> {result!r}")
        return result

    def write_file(self, path: str, content: str) -> str:
        result = self.edit_tool(path, content)
        self.history.append(f"write_file({path!r}) -> {result!r}")
        return result

    def create_file(self, path: str, content: str = "") -> str:
        """Create ``path`` with ``content`` and log the operation."""
        result = self.create_tool(path, content)
        self.history.append(f"create_file({path!r}) -> {result!r}")
        return result

    def delete_file(self, path: str) -> str:
        result = self.delete_tool(path)
        self.history.append(f"delete_file({path!r}) -> {result!r}")
        return result

    def scrape(self, url: str) -> str:
        """Fetch a web page and return plain text."""
        result = self.scrape_tool(url)
        self.history.append(f"scrape({url!r}) -> {result!r}")
        return result

    def run_script(self, path: str) -> str:
        result = self.run_tool(path)
        self.history.append(f"run_script({path!r}) -> {result!r}")
        return result
