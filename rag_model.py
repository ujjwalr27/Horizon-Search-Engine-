import re
from typing import Optional
from logging_config import get_module_logger

class SimplifiedRAG:
    def __init__(self):
        self.logger = get_module_logger('rag')
        
    async def async_generate_response(self, query: str, context: str) -> str:
        """
        Simplified response generation using basic text extraction
        """
        try:
            # Basic text summarization using sentence extraction
            sentences = re.split(r'[.!?]+', context)
            relevant_sentences = [s.strip() for s in sentences if any(q.lower() in s.lower() for q in query.split())]
            
            if relevant_sentences:
                summary = ' '.join(relevant_sentences[:2])  # Take first two relevant sentences
                return summary if len(summary) > 20 else context[:200] + "..."
            return context[:200] + "..."  # Fallback to simple truncation
            
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            return "Unable to generate summary."