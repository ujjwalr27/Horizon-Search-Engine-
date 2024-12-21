import asyncio
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache

class AdvancedFilter:
    def __init__(self, results, logger=None):
        self.filtered = results.copy()
        self.logger = logger or logging.getLogger(__name__)
        self.blacklist_domains = self._load_blacklist()

    def _load_blacklist(self, blacklist_path='blacklist.txt'):
        """
        Load and cache blacklist domains
        """
        try:
            with open(blacklist_path, 'r') as f:
                return set(domain.strip() for domain in f.readlines() if domain.strip())
        except FileNotFoundError:
            self.logger.warning(f"Blacklist file not found: {blacklist_path}")
            return set()

    @lru_cache(maxsize=1000)
    def extract_url_details(self, url):
        """
        Extract and cache URL details with memoization
        """
        try:
            parsed_url = urlparse(url)
            return {
                'domain': parsed_url.hostname,
                'path_length': len(parsed_url.path),
                'is_tracked_domain': parsed_url.hostname in self.blacklist_domains
            }
        except Exception as e:
            self.logger.error(f"URL parsing error: {e}")
            return {'domain': None, 'path_length': 0, 'is_tracked_domain': False}

    async def async_extract_page_details(self, html):
        """
        Asynchronous page detail extraction with robust error handling
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Content analysis
            text_content = soup.get_text(separator=' ', strip=True)
            word_count = len(text_content.split())
            
            # Link and script analysis
            external_links = [
                link.get('href') for link in soup.find_all('a', href=True)
                if not link.get('href', '').startswith('#')
            ]
            
            scripts = [
                script.get('src') for script in soup.find_all('script', src=True)
            ]
            
            return {
                'word_count': word_count,
                'external_link_count': len(external_links),
                'script_count': len(scripts)
            }
        except Exception as e:
            self.logger.error(f"Page extraction error: {e}")
            return {'word_count': 0, 'external_link_count': 0, 'script_count': 0}

    async def parallel_page_analysis(self):
        """
        Perform parallel page analysis using asyncio
        """
        page_details = []
        
        async def process_page(html):
            return await self.async_extract_page_details(html)
        
        tasks = [process_page(html) for html in self.filtered['html']]
        page_details = await asyncio.gather(*tasks)
        
        return pd.DataFrame(page_details)

    def compute_content_score(self, page_details):
        """
        Advanced content scoring mechanism
        """
        word_count_score = np.log(page_details['word_count'] + 1)
        link_score = -0.5 * page_details['external_link_count']
        script_score = -0.3 * page_details['script_count']
        
        return word_count_score + link_score + script_score

    async def filter(self, word_count_threshold=50):
        """
        Comprehensive asynchronous filtering
        """
        try:
            # Parallel page details extraction
            page_details = await self.parallel_page_analysis()
            
            # Compute content scores
            self.filtered['content_score'] = self.compute_content_score(page_details)
            
            # Filter based on content score and word count
            valid_content_mask = page_details['word_count'] >= word_count_threshold
            self.filtered.loc[~valid_content_mask, 'final_rank'] += 10  # Penalize low-content pages
            
            # Final ranking computation
            self.filtered['final_rank'] = (
                self.filtered['rank'] * 0.5 +  # Original ranking weight
                self.filtered['content_score'] * 0.3 +  # Content score weight
                page_details['external_link_count'] * 0.2  # External links influence
            )
            
            # Sort by final rank
            self.filtered = self.filtered.sort_values('final_rank', ascending=True)
            
            return self.filtered
        
        except Exception as e:
            self.logger.error(f"Filtering process error: {e}")
            return self.filtered

# Utility functions for external use
def load_blacklist_domains(path='blacklist.txt'):
    """
    Load blacklist domains for external use
    """
    with open(path, 'r') as f:
        return set(domain.strip() for domain in f.readlines() if domain.strip())
