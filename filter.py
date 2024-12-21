import asyncio
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Set, Dict, Any
from logging_config import get_module_logger

class OptimizedFilter:
    def __init__(self, results: pd.DataFrame):
        self.filtered = results.copy()
        self.logger = get_module_logger('filter')
        self.blacklist_domains = self._load_blacklist()

    def _load_blacklist(self, blacklist_path='blacklist.txt') -> Set[str]:
        """Load blacklist with error handling"""
        try:
            with open(blacklist_path, 'r') as f:
                return {domain.strip() for domain in f if domain.strip()}
        except Exception as e:
            self.logger.error(f"Blacklist loading error: {e}")
            return set()

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc
        except Exception:
            return ""

    def _basic_content_analysis(self, html: str) -> Dict[str, Any]:
        """Simplified content analysis"""
        try:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator=' ', strip=True)
            return {
                'word_count': len(text.split()),
                'link_count': len(soup.find_all('a', href=True))
            }
        except Exception as e:
            self.logger.error(f"Content analysis error: {e}")
            return {'word_count': 0, 'link_count': 0}

    async def filter(self, min_words: int = 50) -> pd.DataFrame:
        """Optimized filtering process"""
        try:
            # Filter blacklisted domains
            domains = self.filtered['link'].apply(self._extract_domain)
            self.filtered = self.filtered[~domains.isin(self.blacklist_domains)].copy()

            # Basic content filtering
            content_scores = []
            for html in self.filtered['html']:
                analysis = self._basic_content_analysis(html)
                score = analysis['word_count'] / max(1, analysis['link_count'])
                content_scores.append(score)

            self.filtered['content_score'] = content_scores
            
            # Apply minimum word count filter
            valid_content = self.filtered['content_score'] >= min_words
            self.filtered = self.filtered[valid_content].copy()

            # Update ranking
            if not self.filtered.empty:
                self.filtered['final_rank'] = range(1, len(self.filtered) + 1)

            return self.filtered

        except Exception as e:
            self.logger.error(f"Filtering error: {e}")
            return self.filtered