import os
from dotenv import load_dotenv

load_dotenv()

# Core Configuration
SEARCH_KEY = os.getenv('GOOGLE_SEARCH_KEY')
SEARCH_ID = os.getenv('GOOGLE_SEARCH_ID')
SEARCH_URL = "https://www.googleapis.com/customsearch/v1?key={key}&cx={cx}&q={query}&start={start}&num=10"

# Database Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Redis Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_MAX_CONNECTIONS = 3
REDIS_TIMEOUT = 2

# Application Settings
MAX_SEARCH_RESULTS = 10
MAX_CONCURRENT_REQUESTS = 5
REQUEST_TIMEOUT = 5
CACHE_EXPIRY = 1800  # 30 minutes
MAX_CACHE_ENTRIES = 100
MAX_CACHE_SIZE = 50000  # ~50KB per entry

# Server Configuration
PORT = int(os.getenv('PORT', 5000))
WORKERS = 1
THREADS = 2