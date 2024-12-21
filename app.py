import os
import asyncio
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Import optimized modules
from logging_config import setup_logging, get_module_logger
from search import OptimizedSearch
from storage import OptimizedDBStorage
from filter import OptimizedFilter
from ml_ranking import SimplifiedMLRanker
from adaptive_cache import RedisAdaptiveCache
from semantic_search import SemanticSearch

# Configure logging
logger = setup_logging()
search_logger = get_module_logger('search')
ml_logger = get_module_logger('ml_ranking')

class OptimizedSearchApp:
    def __init__(self):
        # Initialize Flask App
        self.app = Flask(__name__)
        CORS(self.app)

        # Setup routes
        self.setup_routes()

        # Initialize components with error handling
        try:
            self.db_storage = OptimizedDBStorage()
            self.adaptive_cache = RedisAdaptiveCache()
            self.ml_ranker = SimplifiedMLRanker()
            self.semantic_search = SemanticSearch()
            self.search_engine = OptimizedSearch()

            logger.info("OptimizedSearchApp initialized successfully")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            logger.error(traceback.format_exc())
            raise

    def setup_routes(self):
        """Setup application routes with logging"""
        self.app.route('/', methods=['GET', 'POST'])(self.index)
        self.app.route('/search', methods=['GET'])(self.search_results)
        self.app.route('/mark-relevant', methods=['POST'])(self.mark_relevant)
        self.app.route('/semantic-search', methods=['POST'])(self.perform_semantic_search)

        self.app.errorhandler(404)(self.not_found)
        self.app.errorhandler(500)(self.server_error)

    async def _optimized_search_pipeline(self, query):
        """Optimized search pipeline with error handling"""
        try:
            # Check cache first
            cached_results = await self.adaptive_cache.async_get(query)
            if cached_results:
                return cached_results

            # Perform search
            results = await self.search_engine.search(query)
            if results.empty:
                return []

            # Apply filtering
            filter_obj = OptimizedFilter(results)
            filtered_results = await filter_obj.filter()

            # Apply ML ranking
            try:
                ranked_results = self.ml_ranker.predict_ranking(filtered_results)
            except Exception as e:
                ml_logger.error(f"ML Ranking error: {e}")
                ranked_results = filtered_results

            # Convert to list for caching
            final_results = ranked_results.to_dict('records')

            # Cache results
            await self.adaptive_cache.async_put(query, final_results)

            return final_results

        except Exception as e:
            search_logger.error(f"Search pipeline error: {e}")
            search_logger.error(traceback.format_exc())
            return []

    def search_results(self):
        """Search results route with optimized processing"""
        query = request.args.get('query', '').strip()
        if not query:
            return redirect(url_for('index'))

        try:
            # Run optimized search pipeline
            results = asyncio.run(self._optimized_search_pipeline(query))
            return render_template('results.html', query=query, results=results)
        except Exception as e:
            search_logger.error(f"Search route error: {e}")
            return render_template('error.html', error=str(e)), 500

    def index(self):
        """Optimized home page route"""
        try:
            if request.method == 'POST':
                query = request.form.get('query', '').strip()
                if query:
                    return redirect(url_for('search_results', query=query))
            return render_template('home.html')
        except Exception as e:
            logger.error(f"Index route error: {e}")
            return render_template('error.html', error=str(e)), 500

    def mark_relevant(self):
        """Optimized relevance marking"""
        try:
            data = request.get_json()
            query = data.get('query')
            link = data.get('link')

            if not query or not link:
                return jsonify({'status': 'error', 'message': 'Invalid request'}), 400

            asyncio.run(self.db_storage.update_relevance(query, link, 1.0))
            return jsonify({'status': 'success'})
        except Exception as e:
            logger.error(f"Mark relevant error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def perform_semantic_search(self):
        """Optimized semantic search endpoint"""
        try:
            data = request.json
            query = data.get('query', '')
            documents = data.get('documents', [])

            if not query or not documents:
                return jsonify({"error": "Invalid request"}), 400

            similarities = self.semantic_search.batch_semantic_search(query, documents)
            return jsonify({"similarities": similarities.tolist()})
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return jsonify({"error": str(e)}), 500

    def not_found(self, error):
        return render_template('error.html', error="Page not found"), 404

    def server_error(self, error):
        return render_template('error.html', error="Internal server error"), 500

    def get_app(self):
        return self.app

# Create application instance
app = OptimizedSearchApp().get_app()
main = app

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    main.run(host='0.0.0.0', port=port)
