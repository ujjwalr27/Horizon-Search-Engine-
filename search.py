import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import nltk
from settings import (
    SEARCH_URL, SEARCH_KEY, SEARCH_ID, RESULT_COUNT, 
    MAX_WORKERS, THREAD_POOL_SIZE
)
from storage import DBStorage
from semantic_search import SemanticSearch

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class AdvancedSearch:
    def __init__(self):
        self.semantic_search = SemanticSearch()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_query(self, query):
        # Remove stop words and tokenize
        tokens = word_tokenize(query.lower())
        processed_query = ' '.join([token for token in tokens if token not in self.stop_words])
        return processed_query

    async def fetch_url(self, session, url, timeout=5):
        try:
            async with session.get(url, timeout=timeout) as response:
                return await response.text()
        except Exception as e:
            print(f"Fetch error for {url}: {e}")
            return ""

    async def parallel_scrape(self, links):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, link) for link in links]
            return await asyncio.gather(*tasks)

    async def search_api_async(self, query, pages=3):
        results = []
        async with aiohttp.ClientSession() as session:
            for i in range(pages):
                start = i * 10 + 1
                url = SEARCH_URL.format(
                    key=SEARCH_KEY,
                    cx=SEARCH_ID,
                    query=quote_plus(query),
                    start=start
                )
                async with session.get(url) as response:
                    data = await response.json()
                    results.extend(data.get("items", []))
        
        res_df = pd.DataFrame.from_dict(results)
        res_df["rank"] = list(range(1, res_df.shape[0] + 1))
        res_df = res_df[["link", "rank", "snippet", "title"]]
        return res_df

    async def semantic_search_ranking(self, query, results):
        processed_query = self.preprocess_query(query)
        results["similarity_score"] = await asyncio.to_thread(
            self.semantic_search.batch_semantic_search, 
            processed_query, 
            results["snippet"].tolist()
        )
        results = results.sort_values("similarity_score", ascending=False)
        results["rank"] = list(range(1, results.shape[0] + 1))
        return results

    async def search(self, query):
        columns = ["query", "rank", "link", "title", "snippet", "html", "created", 
                   "semantic_score", "ml_rank", "final_rank"]
        storage = DBStorage()

        # Check if query results are cached
        stored_results = await storage.query_results(query)
        if not stored_results.empty:
            stored_results["created"] = pd.to_datetime(stored_results["created"])
            return stored_results[columns]

        results = await self.search_api_async(query)
        html = await self.parallel_scrape(results["link"])
        results["html"] = html
        results = results[results["html"].str.len() > 0].copy()

        # Semantic search ranking
        results = await self.semantic_search_ranking(query, results)

        results["query"] = query
        results["created"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ensure all required columns are present with default values
        for col in ["semantic_score", "ml_rank", "final_rank"]:
            if col not in results.columns:
                results[col] = 0.0
        
        results = results[columns]

        # Cache the results in the database
        for _, row in results.iterrows():
            await storage.insert_row(row)
        
        return results
