import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import asyncio
import concurrent.futures
from typing import List, Union
from logging_config import get_module_logger

class SemanticSearch:
    """Lightweight semantic search using TF-IDF instead of transformers"""
    def __init__(self, tfidf_features=1000):
        self.logger = get_module_logger('semantic_search')
        
        try:
            # Use TF-IDF only for lightweight semantic search
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=tfidf_features,
                dtype=np.float32
            )
            
            self.tfidf_matrix = None
            self.document_index = None
            self.logger.info("Lightweight SemanticSearch initialized")
        except Exception as e:
            self.logger.error(f"Error initializing SemanticSearch: {e}")
            raise

    def precompute_index(self, documents: List[str]):
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            self.document_index = documents
        except Exception as e:
            self.logger.error(f"Error precomputing index: {e}")
            raise

    def compute_similarity(self, query: str, documents: List[str]) -> np.ndarray:
        """Compute TF-IDF based similarity scores"""
        try:
            if not self.tfidf_vectorizer.vocabulary_:
                self.tfidf_vectorizer.fit(documents)
            
            query_vector = self.tfidf_vectorizer.transform([query])
            doc_vectors = self.tfidf_vectorizer.transform(documents)
            
            # Compute cosine similarity
            similarity = (query_vector @ doc_vectors.T).toarray()[0]
            return similarity
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return np.zeros(len(documents))

    def batch_semantic_search(self, query: str, documents: List[str], top_k: int = 5) -> np.ndarray:
        """Perform semantic search using TF-IDF similarity"""
        try:
            return self.compute_similarity(query, documents)
        except Exception as e:
            self.logger.error(f"Error in batch search: {e}")
            return np.zeros(len(documents))

    async def async_semantic_search(self, query: str, documents: List[str], top_k: int = 5) -> np.ndarray:
        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                return await loop.run_in_executor(
                    pool,
                    self.batch_semantic_search,
                    query,
                    documents,
                    top_k
                )
        except Exception as e:
            self.logger.error(f"Error in async search: {e}")
            return np.zeros(len(documents))