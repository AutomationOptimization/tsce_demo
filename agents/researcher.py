import requests
import openai

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


class SearchTool:
    """Simple web search tool using DuckDuckGo."""

    def __call__(self, query: str) -> str:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_redirect": 1}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            return data.get("AbstractText") or data.get("Answer") or ""
        except Exception as exc:  # pragma: no cover - network issues
            return f"Search error: {exc}"


class FileTool:
    """Basic file read/write helpers."""

    def read(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def write(self, path: str, content: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return "ok"


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
        self.search_tool = SearchTool()
        self.file_tool = FileTool()

    # Convenience wrappers -------------------------------------------------
    def search(self, query: str) -> str:
        return self.search_tool(query)

    def read_file(self, path: str) -> str:
        return self.file_tool.read(path)

    def write_file(self, path: str, content: str) -> str:
        return self.file_tool.write(path, content)
