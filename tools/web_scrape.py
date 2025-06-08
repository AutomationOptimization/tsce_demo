import re
import requests


def _strip_tags(html: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class WebScrape:
    """Fetch a web page and return plain text."""

    def __call__(self, url: str) -> str:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - network issues
            return f"Scrape error: {exc}"

        return _strip_tags(resp.text)
