from supabase import create_client, Client
import pandas as pd
from datetime import datetime
import asyncio
from typing import Dict, Any, Optional
from settings import SUPABASE_URL, SUPABASE_KEY
from logging_config import get_module_logger

class OptimizedDBStorage:
    def __init__(self):
        self.logger = get_module_logger('storage')
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    async def query_results(self, query: str) -> pd.DataFrame:
        """Optimized query results retrieval"""
        try:
            response = self.supabase.table('results')\
                .select('*')\
                .eq('query', query)\
                .order('final_rank')\
                .limit(20)\
                .execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            self.logger.error(f"Query error: {e}")
            return pd.DataFrame()

    async def insert_row(self, values: Dict[str, Any]) -> Optional[Dict]:
        """Optimized row insertion"""
        try:
            row_data = {
                'query': values.get('query', ''),
                'rank': values.get('rank', 0),
                'link': values.get('link', ''),
                'title': values.get('title', ''),
                'snippet': values.get('snippet', ''),
                'created': datetime.utcnow().isoformat()
            }
            response = self.supabase.table('results').upsert(row_data).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Insert error: {e}")
            return None

    async def update_relevance(self, query: str, link: str, relevance: float) -> Optional[Dict]:
        """Optimized relevance update"""
        try:
            response = self.supabase.table('results')\
                .update({'relevance': relevance})\
                .eq('query', query)\
                .eq('link', link)\
                .execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Update error: {e}")
            return None