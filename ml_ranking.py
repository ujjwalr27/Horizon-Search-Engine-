import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List
from logging_config import get_module_logger

class SimplifiedMLRanker:
    def __init__(self):
        self.logger = get_module_logger('ml_ranking')
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            dtype=np.float32
        )
        
    def predict_ranking(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Simplified ranking using TF-IDF similarity
        """
        try:
            if results.empty:
                return results
                
            # Combine title and snippet for ranking
            text_data = results['title'].fillna('') + ' ' + results['snippet'].fillna('')
            
            # Calculate TF-IDF scores
            tfidf_matrix = self.vectorizer.fit_transform(text_data)
            
            # Calculate document importance scores
            importance_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
            
            # Normalize scores
            if len(importance_scores) > 0:
                min_score = importance_scores.min()
                max_score = importance_scores.max()
                if max_score > min_score:
                    normalized_scores = (importance_scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores = np.ones_like(importance_scores)
            else:
                normalized_scores = np.ones_like(importance_scores)
            
            # Add scores to results
            results['ml_rank'] = normalized_scores
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ranking error: {e}")
            results['ml_rank'] = 1.0
            return results