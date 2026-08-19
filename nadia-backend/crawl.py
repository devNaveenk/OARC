"""
One-off crawler: builds knowledge_base.jsonl for the Nadia chatbot
by fetching every URL in cityofwinterpark.org's sitemap and extracting
clean page text. Run once (or re-run to refresh the knowledge base).
"""
import json
import re
import time
import sys
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NadiaKBBot/1.0; +https://cityofwinterpark.org)"}
TIMEOUT = 15

def extract_text(html: str, url: str):
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return title, ""

    text_parts = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "caption", "figcaption", "span"]):
        t = el.get_text(" ", strip=True)
        if t and len(t) > 1 and (not text_parts or text_parts[-1] != t):
            text_parts.append(t)

    for a in main.find_all("a", href=True):
        href = a["href"]
        if href.startswith("mailto:") or href.startswith("tel:"):
            label = a.get_text(" ", strip=True) or href.split(":", 1)[1]
            contact_value = href.split(":", 1)[1]
            text_parts.append(f"{label}: {contact_value}")

    text = "\n".join(text_parts)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return title, text.strip()


def main():
    with open("crawl_urls.txt") as f:
        urls = [u.strip() for u in f if u.strip()]

    out = open("knowledge_base.jsonl", "w", encoding="utf-8")
    ok, fail = 0, 0
    for i, url in enumerate(urls, 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code != 200:
                fail += 1
                print(f"[{i}/{len(urls)}] SKIP {r.status_code} {url}", file=sys.stderr)
                continue
            title, text = extract_text(r.text, url)
            if len(text) < 8:
                fail += 1
                continue
            out.write(json.dumps({"url": url, "title": title, "text": text}, ensure_ascii=False) + "\n")
            ok += 1
            if i % 25 == 0:
                print(f"[{i}/{len(urls)}] ok={ok} fail={fail}", file=sys.stderr)
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(urls)}] ERROR {url}: {e}", file=sys.stderr)
        time.sleep(0.15)

    out.close()
    print(f"DONE ok={ok} fail={fail}", file=sys.stderr)


if __name__ == "__main__":
    main()
