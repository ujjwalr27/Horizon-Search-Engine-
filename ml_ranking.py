import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import asyncio
import concurrent.futures
from settings import ML_MODEL_PATH

class AdvancedMLRanker:
    def __init__(self):
        self.text_features = ['title', 'snippet']
        self.numeric_features = ['link_length', 'title_length']
        
        # Advanced preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(
                    stop_words='english', 
                    max_features=1000, 
                    ngram_range=(1, 2)
                ), self.text_features),
                ('numeric', StandardScaler(), self.numeric_features)
            ])
        
        # Ensemble of multiple models
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200, 
                max_depth=10, 
                random_state=42
            ))
        ])

    def _validate_and_prepare_data(self, results):
        """
        Validate and prepare training data
        """
        results = results.copy()
        
        # Default handling for missing columns
        for col in self.text_features + self.numeric_features + ['relevance']:
            if col not in results.columns:
                results[col] = 0 if col != 'relevance' else np.nan

        # Handle missing relevance values
        results['relevance'] = pd.to_numeric(results['relevance'], errors='coerce').fillna(0)
        
        # Compute additional features
        results['link_length'] = results['link'].str.len()
        results['title_length'] = results['title'].str.len()
        
        return results

    def train(self, results):
        """
        Train ML model with advanced preprocessing
        """
        try:
            results = self._validate_and_prepare_data(results)
            
            X = results[self.text_features + self.numeric_features]
            y = results['relevance']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Fit model
            self.model.fit(X_train, y_train)

            # Model evaluation
            y_pred = self.model.predict(X_test)
            print(f"Model Performance:")
            print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
            print(f"R² Score: {r2_score(y_test, y_pred)}")

            return self.model
        except Exception as e:
            print(f"Training error: {e}")
            return None

    async def async_train(self, results):
        """
        Asynchronous training wrapper
        """
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self.train, results)

    def predict_ranking(self, results):
        """
        Predict rankings with confidence scoring
        """
        try:
            results = self._validate_and_prepare_data(results)
            
            # Predict probabilities for ranking
            predictions = self.model.predict(results[self.text_features + self.numeric_features])
            
            # Normalize predictions
            results['ml_rank'] = (predictions - predictions.min()) / (predictions.max() - predictions.min())
            
            return results
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback ranking
            results['ml_rank'] = np.linspace(0, 1, len(results))
            return results

    def save_model(self, path=ML_MODEL_PATH):
        """
        Save entire model pipeline
        """
        joblib.dump(self.model, path)

    def load_model(self, path=ML_MODEL_PATH):
        """
        Load entire model pipeline
        """
        self.model = joblib.load(path)
