import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Search Configuration
SEARCH_KEY = os.getenv('GOOGLE_SEARCH_KEY')
SEARCH_ID = os.getenv('GOOGLE_SEARCH_ID')
COUNTRY = "in"
SEARCH_URL = "https://www.googleapis.com/customsearch/v1?key={key}&cx={cx}&q={query}&start={start}&num=10&gl=" + COUNTRY
RESULT_COUNT = 30  # Increased for better results

# Parallel Processing Configuration
MAX_WORKERS = 10
THREAD_POOL_SIZE = 20
PORT = os.getenv('PORT', 5000)

# Caching Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
CACHE_EXPIRY = int(os.getenv('CACHE_EXPIRY', 3600))
LRU_CACHE_SIZE = int(os.getenv('LRU_CACHE_SIZE', 1000))

# ML and AI Configuration
ML_MODEL_PATH = 'ml_ranker.joblib'
RAG_MODEL = 'facebook/bart-small'
SUMMARIZATION_MODEL = 'paraphrase-MiniLM-L3-v2'

# Advanced Query Preprocessing
STOP_WORDS_FILE = 'stop_words.txt'
