import torch
from typing import Optional
import faiss
from contextlib import contextmanager
import pickle
from transformers import (
    DPRContextEncoder, 
    DPRQuestionEncoder, 
    DPRContextEncoderTokenizer, 
    DPRQuestionEncoderTokenizer
)

class ModelManager:
    _instance: Optional['ModelManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._question_encoder = None
            self._context_encoder = None
            self._question_tokenizer = None
            self._context_tokenizer = None
            self._initialized = True
    
    @property
    def question_encoder(self):
        if self._question_encoder is None:
            self._question_encoder = DPRQuestionEncoder.from_pretrained(
                'facebook/dpr-question_encoder-single-nq-base'
            ).to(self.device)
        return self._question_encoder
    
    @property
    def context_encoder(self):
        if self._context_encoder is None:
            self._context_encoder = DPRContextEncoder.from_pretrained(
                'facebook/dpr-ctx_encoder-single-nq-base'
            ).to(self.device)
        return self._context_encoder
    
    @property
    def question_tokenizer(self):
        if self._question_tokenizer is None:
            self._question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                'facebook/dpr-question_encoder-single-nq-base'
            )
        return self._question_tokenizer
    
    @property
    def context_tokenizer(self):
        if self._context_tokenizer is None:
            self._context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                'facebook/dpr-ctx_encoder-single-nq-base'
            )
        return self._context_tokenizer
    
    def cleanup(self):
        if self._question_encoder is not None:
            self._question_encoder.cpu()
            del self._question_encoder
            self._question_encoder = None
            
        if self._context_encoder is not None:
            self._context_encoder.cpu()
            del self._context_encoder
            self._context_encoder = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class FAISSIndexManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FAISSIndexManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.index = None
            self.chunk_data = []
            self._initialized = True
    
    def load_index(self, index_path: str):
        self.index = faiss.read_index(index_path)
        with open(f"{index_path}_metadata.pkl", 'rb') as f:
            self.chunk_data = pickle.load(f)
            
    def cleanup(self):
        if self.index is not None:
            self.index.reset()
            self.index = None
        self.chunk_data = []

@contextmanager
def cuda_memory_manager():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()