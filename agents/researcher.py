import requests
import openai
from typing import List

from tools import (
    GoogleSearch,
    WebScrape,
    CreateFileTool,
    ReadFileTool,
    EditFileTool,
    DeleteFileTool,
    RunScriptTool,
)

class BaseAgent:
    """Minimal base agent that sends prompts to an OpenAI compatible model."""

    def __init__(self, name: str, system_message: str = "", model: str = "gpt-3.5-turbo") -> None:
        self.name = name
        self.system_message = system_message
        self.model = model

    def chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        resp = openai.ChatCompletion.create(model=self.model, messages=messages)
        return resp["choices"][0]["message"]["content"]

class Researcher(BaseAgent):
    """Agent capable of searching the web and reading/writing files."""

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        super().__init__(
            name="Researcher",
            system_message=(
                "You are a meticulous research assistant. "
                "Use your search and file tools when helpful."
            ),
            model=model,
        )
        self.history: List[str] = []
        self.search_tool = GoogleSearch()
        self.scrape_tool = WebScrape()
        self.create_tool = CreateFileTool()
        self.read_tool = ReadFileTool()
        self.edit_tool = EditFileTool()
        self.delete_tool = DeleteFileTool()
        self.run_tool = RunScriptTool()

    # Convenience wrappers -------------------------------------------------
    def search(self, query: str) -> str:
        """Return top Google results for ``query`` and log the call."""
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
