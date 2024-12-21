import os
import sys
import traceback
import asyncio
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom modules with logging
from logging_config import setup_logging, get_module_logger
from search import AdvancedSearch
from storage import DBStorage
from filter import AdvancedFilter
from ml_ranking import AdvancedMLRanker
from adaptive_cache import RedisAdaptiveCache
from rag_model import AdvancedRAG
from semantic_search import SemanticSearch

# Configure logging
logger = setup_logging()
search_logger = get_module_logger('search')
ml_logger = get_module_logger('ml_ranking')

class HorizonSearchApp:
    def __init__(self):
        # Initialize Flask App
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS

        # Setup routes
        self.setup_routes()

        # Initialize global components
        try:
            self.db_storage = DBStorage()
            self.adaptive_cache = RedisAdaptiveCache()
            self.ml_ranker = AdvancedMLRanker()
            self.rag_model = AdvancedRAG()
            self.semantic_search = SemanticSearch()
            self.advanced_search = AdvancedSearch()

            # Background tasks
            self._load_ml_model()
            self._setup_background_tasks()

            logger.info("HorizonSearchApp initialized successfully")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            logger.error(traceback.format_exc())
            raise

    def setup_routes(self):
        """
        Setup application routes with logging
        """
        self.app.route('/', methods=['GET', 'POST'])(self.index)
        self.app.route('/search', methods=['GET'])(self.search_results)
        self.app.route('/mark-relevant', methods=['POST'])(self.mark_relevant)
        self.app.route('/semantic-search', methods=['POST'])(self.perform_semantic_search)

        self.app.errorhandler(404)(self.not_found)
        self.app.errorhandler(500)(self.server_error)

    async def _advanced_search_pipeline(self, query):
        """
        Comprehensive async search pipeline
        """
        try:
            # Perform initial search
            results = await self.advanced_search.search(query)

            # Apply advanced filtering with logging
            filter_obj = AdvancedFilter(results, logger=search_logger)
            filtered_results = await filter_obj.filter()

            # ML Ranking with error handling
            try:
                ml_ranked_results = self.ml_ranker.predict_ranking(filtered_results)
            except Exception as e:
                ml_logger.error(f"ML Ranking error: {e}")
                ml_ranked_results = filtered_results

            # RAG summary generation
            final_results = ml_ranked_results.to_dict('records')
            for result in final_results:
                try:
                    result['rag_summary'] = await self.rag_model.async_generate_response(
                        query, 
                        result.get('snippet', '')
                    )
                except Exception as e:
                    search_logger.error(f"RAG summary error: {e}")
                    result['rag_summary'] = "Summary generation failed."

            return final_results
        
        except Exception as e:
            search_logger.error(f"Search pipeline error: {e}")
            search_logger.error(traceback.format_exc())
            return []

    def search_results(self):
        """
        Search results route with advanced async processing
        """
        query = request.args.get('query', '').strip()

        if not query:
            return redirect(url_for('index'))

        try:
            # Check cache first
            cached_results = self.adaptive_cache.get(query)
            if cached_results:
                return render_template('results.html', query=query, results=cached_results)

            # Run async search pipeline
            final_results = asyncio.run(self._advanced_search_pipeline(query))

            # Cache the results
            self.adaptive_cache.put(query, final_results)

            return render_template('results.html', query=query, results=final_results)

        except Exception as e:
            search_logger.error(f"Search route error: {e}")
            return render_template('error.html', error=str(e)), 500

    def index(self):
        """
        Home page route with logging
        """
        try:
            if request.method == 'POST':
                query = request.form.get('query', '').strip()
                if query:
                    logger.info(f"Search query received: {query}")
                    return redirect(url_for('search_results', query=query))
            return render_template('home.html')
        except Exception as e:
            logger.error(f"Index route error: {e}")
            return render_template('error.html', error=str(e)), 500

    def mark_relevant(self):
        """
        Mark a result as relevant
        """
        try:
            data = request.get_json()
            query = data.get('query')
            link = data.get('link')

            if not query or not link:
                return jsonify({'status': 'error', 'message': 'Invalid request'}), 400

            # Use the DBStorage to collect user feedback
            asyncio.run(self.db_storage.collect_user_feedback(query, link, 1.0))

            # Update the relevance in the database
            asyncio.run(self.db_storage.update_relevance(query, link, 1.0))

            return jsonify({'status': 'success', 'message': 'Relevance marked'})
        except Exception as e:
            logger.error(f"Mark relevant error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def perform_semantic_search(self):
        """
        Perform semantic search
        """
        try:
            data = request.json
            query = data.get('query', '')
            documents = data.get('documents', [])

            if not query or not documents:
                return jsonify({"error": "Query and documents are required"}), 400

            # Perform semantic search
            similarities = self.semantic_search.semantic_search(query, documents)

            return jsonify({"similarities": similarities.tolist()})

        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return jsonify({"error": "An error occurred during semantic search", "details": str(e)}), 500

    def not_found(self, error):
        return render_template('error.html', error="Page not found"), 404

    def server_error(self, error):
        return render_template('error.html', error="Internal server error"), 500

    def _load_ml_model(self):
        """Load ML models in the background"""
        logger.info("Loading ML models...")

    def _setup_background_tasks(self):
        """Initialize background tasks"""
        logger.info("Setting up background tasks...")

    def get_app(self):
        """Return the Flask application instance"""
        return self.app

# Create the application instance
app = HorizonSearchApp().get_app()

# This is the WSGI application callable
main = app

if __name__ == '__main__':
   port = int(os.getenv('PORT', 5000))
   main.run(host='0.0.0.0', port=port)