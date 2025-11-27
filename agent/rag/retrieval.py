import os
import re
from typing import List, Dict
from rank_bm25 import BM25Okapi

class SimpleRetriever:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.chunks = []
        self.bm25 = None
        self._load_and_index()

    def _tokenize(self, text: str) -> List[str]:
        # Better tokenization: alphanumeric only, lowercase
        return re.findall(r'\b\w+\b', text.lower())

    def _load_and_index(self):
        self.chunks = []
        tokenized_corpus = []

        for filename in os.listdir(self.docs_dir):
            if filename.endswith(".md"):
                filepath = os.path.join(self.docs_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Simple splitting by headers or double newlines
                raw_chunks = re.split(r'\n#{1,3} |\n\n', content)
                
                for i, text in enumerate(raw_chunks):
                    if text.strip():
                        chunk_id = f"{filename}::chunk{i}"
                        self.chunks.append({
                            "id": chunk_id,
                            "content": text.strip(),
                            "source": filename
                        })
                        # Improved tokenization
                        tokenized_corpus.append(self._tokenize(text))
        
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        if not self.bm25:
            return []
            
        tokenized_query = self._tokenize(query)
        # Get top k scores
        scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_n_indices:
            # Arbitrary score threshold could be added here
            results.append({
                **self.chunks[idx],
                "score": scores[idx]
            })
            
        return results

