import os
import sys
import requests
import json
from bs4 import BeautifulSoup
from openai import AzureOpenAI

def scrape_webpage(url, timeout=15):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    return " ".join(soup.stripped_strings)

def get_azure_client():
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_KEY_C"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_C"],
    )

def summarize_content(client, deployment, text, url):
    prompt = (
        f"Summarize the content of the following webpage ({url}) in a concise paragraph:\n\n{text[:8000]}"
    )
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_webpage.py <URL>")
        sys.exit(1)
    url = sys.argv[1]
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_C", "gpt-4o")
    client = get_azure_client()
    text = scrape_webpage(url)
    summary = summarize_content(client, deployment, text, url)
    print(json.dumps({
        "url": url,
        "summary": summary
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()