import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from urllib.parse import quote_plus
from typing import List, Dict, Optional
from logging_config import get_module_logger
from settings import SEARCH_URL, SEARCH_KEY, SEARCH_ID

class OptimizedSearch:
    def __init__(self):
        self.logger = get_module_logger('search')
        
    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch URL with timeout and error handling"""
        try:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    return await response.text()
                return ""
        except Exception as e:
            self.logger.error(f"Fetch error for {url}: {e}")
            return ""

    async def parallel_fetch(self, urls: List[str], max_concurrent: int = 5) -> List[str]:
        """Fetch URLs in parallel with concurrency limit"""
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def fetch_with_semaphore(url: str) -> str:
                async with semaphore:
                    return await self.fetch_url(session, url)
            
            tasks = [fetch_with_semaphore(url) for url in urls]
            return await asyncio.gather(*tasks)

    async def search_api(self, query: str, max_results: int = 10) -> pd.DataFrame:
        """Search API with optimized result handling"""
        try:
            url = SEARCH_URL.format(
                key=SEARCH_KEY,
                cx=SEARCH_ID,
                query=quote_plus(query),
                start=1
            )
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    items = data.get("items", [])[:max_results]
                    
            if not items:
                return pd.DataFrame()
                
            results = pd.DataFrame(items)
            results = results[["link", "snippet", "title"]].copy()
            results["rank"] = range(1, len(results) + 1)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search API error: {e}")
            return pd.DataFrame()

    async def search(self, query: str) -> pd.DataFrame:
        """Main search function with optimizations"""
        try:
            # Get initial results
            results = await self.search_api(query)
            if results.empty:
                return results
                
            # Fetch HTML in parallel with limited concurrency
            html_contents = await self.parallel_fetch(results["link"].tolist(), max_concurrent=5)
            results["html"] = html_contents
            
            # Filter out failed fetches
            results = results[results["html"].str.len() > 0].copy()
            
            # Add metadata
            results["query"] = query
            results["created"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            results["semantic_score"] = 0.0
            results["ml_rank"] = 0.0
            results["final_rank"] = results["rank"]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return pd.DataFrame()