import re
import requests


class GoogleSearch:
    """Lightweight wrapper around Google search."""

    def __call__(self, query: str, num_results: int = 5) -> list[str]:
        url = "https://www.google.com/search"
        try:
            resp = requests.get(
                url,
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5,
            )
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - network issues
            return [f"Search error: {exc}"]

        titles = re.findall(r"<h3[^>]*>(.+?)</h3>", resp.text, re.DOTALL)
        clean = [re.sub(r"<.*?>", "", t).strip() for t in titles]
        return clean[:num_results]
