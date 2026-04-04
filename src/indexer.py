import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import re


class HybridIndexer:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.bm25 = None
        self.faiss_index = None

    def build_index(self, chunks_path: str):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)


        texts = [c["text"] for c in self.chunks]

        tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)

        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)

        self._save()
        print("Index built and saved!")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        bm25_results = self._bm25_search(query, top_k)
        faiss_results = self._faiss_search(query, top_k)

        combined = self._reciprocal_rank_fusion(bm25_results, faiss_results)
        return combined[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _faiss_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        return [(int(i), float(distances[0][j])) for j, i in enumerate(indices[0])]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        faiss_results: List[Tuple[int, float]],
        k: int = 60) -> List[Dict]:

        scores = {}

        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

        for rank, (idx, _) in enumerate(faiss_results):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in sorted_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = scores[idx]
            results.append(chunk)

        return results

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _save(self):
        out = Path("outputs")
        faiss.write_index(self.faiss_index, str(out / "faiss.index"))
        with open(out / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(out / "chunks.json", "w") as f:
            json.dump(self.chunks, f)

    def load(self):
        out = Path("outputs")
        self.faiss_index = faiss.read_index(str(out / "faiss.index"))
        with open(out / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(out / "chunks.json") as f:
            self.chunks = json.load(f)
        print("Index loaded!")


