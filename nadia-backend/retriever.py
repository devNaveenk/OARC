"""
Lightweight TF-IDF retriever over the scraped city knowledge base.
No external embedding API needed — keeps the whole pipeline local/free
except for the actual DeepSeek chat completion call.
"""
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


class KnowledgeBase:
    def __init__(self, jsonl_path: str):
        self.chunks = []  # list of {"text", "title", "url"}
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                for c in _chunk(doc["text"]):
                    self.chunks.append({"text": c, "title": doc["title"], "url": doc["url"]})

        corpus = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def search(self, query: str, top_k: int = 5):
        if self.matrix is None:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            if sims[idx] <= 0.03:
                continue
            c = self.chunks[idx]
            results.append({"text": c["text"], "title": c["title"], "url": c["url"], "score": float(sims[idx])})
        return results
