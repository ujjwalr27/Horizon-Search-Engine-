from supabase import create_client, Client
import pandas as pd
from datetime import datetime
import asyncio
from settings import SUPABASE_URL, SUPABASE_KEY

class DBStorage:
    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.setup_tables()

    def setup_tables(self):
        # Supabase tables are pre-created in the database
        pass

    async def query_results(self, query):
        try:
            response = self.supabase.table('results').select('*').eq('query', query).order('final_rank').execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            print(f"Query results error: {e}")
            return pd.DataFrame()

    async def insert_row(self, values):
        try:
            row_data = {
                'query': values.get('query', ''),
                'rank': values.get('rank', 0),
                'link': values.get('link', ''),
                'title': values.get('title', ''),
                'snippet': values.get('snippet', ''),
                'html': values.get('html', ''),
                'created': values.get('created', datetime.utcnow().isoformat()),
                'semantic_score': values.get('similarity_score', 0.0),
                'ml_rank': values.get('ml_rank', 0.0),
                'final_rank': values.get('final_rank', values.get('rank', 0))
            }
            
            response = self.supabase.table('results').upsert(row_data).execute()
            return response
        except Exception as e:
            print(f"Insert row error: {e}")

    async def update_relevance(self, query, link, relevance):
        try:
            response = (
                self.supabase.table('results')
                .update({'relevance': relevance})
                .eq('query', query)
                .eq('link', link)
                .execute()
            )
            return response
        except Exception as e:
            print(f"Relevance update error: {e}")

    async def get_training_data(self):
        try:
            response = (
                self.supabase.table('results')
                .select('*')
                .gt('relevance', 0)
                .limit(1000)
                .execute()
            )
            return pd.DataFrame(response.data)
        except Exception as e:
            print(f"Training data retrieval error: {e}")
            return pd.DataFrame()

    async def collect_user_feedback(self, query, link, relevance_score):
        try:
            feedback_data = {
                'query': query,
                'link': link,
                'relevance_score': relevance_score,
                'timestamp': datetime.utcnow().isoformat()
            }
            response = self.supabase.table('user_feedback').upsert(feedback_data).execute()
            return response
        except Exception as e:
            print(f"Feedback collection error: {e}")

    def close(self):
        # Supabase doesn't require explicit connection closing
        pass