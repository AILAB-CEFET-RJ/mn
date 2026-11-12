"""Serviço de baseline léxico Okapi BM25 no nível de objeto (`item_id`)."""
import math
import re
import unicodedata
from collections import Counter

PT_STOPWORDS = {
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos", "e", "em",
    "entre", "era", "essa", "esse", "esta", "este", "foi", "mais", "mas", "na", "nas",
    "no", "nos", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por",
    "qual", "que", "se", "sem", "ser", "seu", "seus", "sob", "sobre", "sua", "suas",
    "sao", "tambem", "tem", "um", "uma", "umas", "uns", "utilizado", "utilizados",
    "utilizada", "utilizadas", "usado", "usada", "usados", "usadas", "objeto", "objetos",
}


def _tokenize_pt(text: str) -> list[str]:
    norm = unicodedata.normalize("NFKD", (text or "").lower())
    ascii_text = "".join(c for c in norm if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]{2,}", ascii_text)
    # Stemming leve de plural em português para tornar o baseline léxico justo
    stemmed = [t[:-1] if len(t) > 4 and t.endswith("s") else t for t in tokens if t not in PT_STOPWORDS]
    return stemmed


class BM25Index:
    """Baseline léxico Okapi BM25 no nível de objeto (`item_id`)."""

    def __init__(self, docs_by_item: dict[str, dict], text_field: str, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.items = list(docs_by_item.values())
        self.tokens = [_tokenize_pt(it.get(text_field) or "") for it in self.items]
        self.doc_lens = [len(t) for t in self.tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        self.df: Counter = Counter()
        for t in self.tokens:
            for term in set(t):
                self.df[term] += 1
        self.N = len(self.items)

    def search(self, query: str, top_k: int = 10, filters: dict | None = None) -> list[tuple[dict, float]]:
        q_terms = _tokenize_pt(query)
        scored: list[tuple[dict, float]] = []
        for idx, it in enumerate(self.items):
            if filters and any(it.get(k) != v for k, v in filters.items()):
                continue
            tf = Counter(self.tokens[idx])
            dl = self.doc_lens[idx]
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                n_q = self.df[term]
                idf = math.log(1.0 + (self.N - n_q + 0.5) / (n_q + 0.5))
                f = tf[term]
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                score += idf * (f * (self.k1 + 1.0)) / denom
            if score > 0:
                scored.append((it, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

