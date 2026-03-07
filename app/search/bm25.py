import math
import re
from collections import Counter

# standard BM25 params
K1 = 1.5
B = 0.75

# words that show up everywhere and hurt retrieval
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "not", "no", "so", "if", "as", "its", "all", "any", "each",
    "than", "then", "too", "very", "just", "about", "up", "out", "into",
    "over", "after", "before", "between", "under", "such", "what", "which",
    "who", "whom", "how", "when", "where", "why", "there", "here", "also",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    """Chunk dicts with text + id. build() then search()."""

    def __init__(self):
        self.docs: list[dict] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: dict[str, int] = {}  # how many docs contain each term
        self.avg_dl: float = 0.0
        self.n_docs: int = 0

    def build(self, chunks: list[dict]):
        self.docs = chunks
        self.doc_tokens = []
        self.doc_freqs = {}
        self.n_docs = len(chunks)

        total_len = 0

        for chunk in chunks:
            tokens = tokenize(chunk["text"])
            self.doc_tokens.append(tokens)
            total_len += len(tokens)

            # count unique terms per doc for IDF
            seen = set(tokens)
            for term in seen:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_dl = total_len / self.n_docs if self.n_docs > 0 else 1.0

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.n_docs == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for i in range(self.n_docs):
            score = self._score_doc(i, query_tokens)
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            chunk = self.docs[idx].copy()
            chunk["bm25_score"] = score
            results.append(chunk)

        return results

    def _score_doc(self, doc_idx: int, query_tokens: list[str]) -> float:
        doc_toks = self.doc_tokens[doc_idx]
        dl = len(doc_toks)
        tf_map = Counter(doc_toks)

        score = 0.0
        for qt in query_tokens:
            if qt not in self.doc_freqs:
                continue

            df = self.doc_freqs[qt]
            tf = tf_map.get(qt, 0)

            # IDF component: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

            # TF component with length normalization
            numerator = tf * (K1 + 1)
            denominator = tf + K1 * (1 - B + B * dl / self.avg_dl)
            score += idf * (numerator / denominator)

        return score


# singleton, gets rebuilt whenever new docs are ingested
bm25_index = BM25Index()
