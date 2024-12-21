import torch
import asyncio
import concurrent.futures
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    AutoModelForSeq2SeqLM,
    pipeline
)
from settings import RAG_MODEL, SUMMARIZATION_MODEL

class AdvancedRAG:
    def __init__(self):
        self.device = 'cpu'  # Force CPU usage
        self.initialized = False
        self.qa_tokenizer = None
        self.qa_model = None
        self.summarization_tokenizer = None
        self.summarization_model = None
        self.qa_pipeline = None
        self.summarization_pipeline = None

    async def initialize(self):
        """Lazy initialization of models"""
        if not self.initialized:
            # QA Model - Using a smaller model
            self.qa_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
            
            # Summarization Model - Using a smaller model
            self.summarization_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
            self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')
            
            # Create pipelines with memory optimization
            self.qa_pipeline = pipeline(
                'question-answering', 
                model=self.qa_model, 
                tokenizer=self.qa_tokenizer,
                device=-1  # Force CPU
            )
            
            self.summarization_pipeline = pipeline(
                'summarization', 
                model=self.summarization_model, 
                tokenizer=self.summarization_tokenizer,
                device=-1  # Force CPU
            )
            
            self.initialized = True

    def _preprocess_context(self, context: str, max_length: int = 384) -> str:
        """
        Preprocess and truncate context to fit smaller model
        """
        if not self.initialized:
            asyncio.run(self.initialize())
        
        tokens = self.qa_tokenizer.encode(context)
        return self.qa_tokenizer.decode(tokens[:max_length])

    async def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using memory-optimized approach
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Preprocess context
            processed_context = self._preprocess_context(context)
            
            # Try Question Answering first
            qa_result = self.qa_pipeline({
                'question': query,
                'context': processed_context
            }, max_length=384)
            
            if qa_result['score'] > 0.5:
                return qa_result['answer']
            
            # Fallback to summarization
            return await self._generate_summary(query, processed_context)
        
        except Exception as e:
            print(f"RAG response generation error: {e}")
            return "Unable to generate a summary."

    async def _generate_summary(self, query: str, context: str) -> str:
        """
        Generate a memory-efficient summary
        """
        try:
            input_text = f"Query: {query}\nContext: {context}"
            summary = self.summarization_pipeline(
                input_text,
                max_length=60,
                min_length=20,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Summarization error: {e}")
            return "Unable to generate a summary."

    def cleanup(self):
        """
        Clean up resources
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()