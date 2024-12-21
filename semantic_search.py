import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import asyncio
import concurrent.futures
from typing import List, Union

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2', tfidf_features=1000):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Sentence Transformer for semantic embeddings
        self.transformer_model = SentenceTransformer(model_name).to(self.device)
        
        # TF-IDF for sparse matrix indexing
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=tfidf_features
        )
        
        # Pre-computed index (to be populated)
        self.tfidf_matrix = None
        self.document_index = None

    def precompute_index(self, documents: List[str]):
        """
        Pre-compute TF-IDF matrix and document index for fast retrieval
        """
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        self.document_index = documents

    def embed_query(self, query: str) -> torch.Tensor:
        """
        Embed a single query using sentence transformer
        """
        return self.transformer_model.encode(
            query, 
            convert_to_tensor=True, 
            show_progress_bar=False
        )

    def embed_documents(self, documents: List[str]) -> torch.Tensor:
        """
        Embed multiple documents in parallel
        """
        return self.transformer_model.encode(
            documents, 
            convert_to_tensor=True, 
            show_progress_bar=False
        )

    def batch_semantic_search(self, query: str, documents: List[str], top_k: int = 5) -> np.ndarray:
        """
        Perform batch semantic search with multiple similarity strategies
        """
        query_embedding = self.embed_query(query)
        doc_embeddings = self.embed_documents(documents)

        # Compute semantic similarity (cosine)
        cos_scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), 
            doc_embeddings
        ).cpu().numpy()

        return cos_scores

    async def async_semantic_search(self, query: str, documents: List[str], top_k: int = 5) -> np.ndarray:
        """
        Asynchronous semantic search wrapper
        """
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(
                pool, 
                self.batch_semantic_search, 
                query, 
                documents, 
                top_k
            )

    def hybrid_search(self, query: str, documents: List[str]) -> List[float]:
        """
        Combine TF-IDF and semantic embedding similarities
        """
        if self.tfidf_matrix is None:
            self.precompute_index(documents)

        # TF-IDF similarity
        query_vector = self.tfidf_vectorizer.transform([query])
        tfidf_scores = np.array((query_vector @ self.tfidf_matrix.T).todense()).flatten()

        # Semantic embedding similarity
        semantic_scores = self.batch_semantic_search(query, documents)

        # Hybrid scoring (weighted combination)
        hybrid_scores = 0.6 * semantic_scores + 0.4 * tfidf_scores
        return hybrid_scores.tolist()
